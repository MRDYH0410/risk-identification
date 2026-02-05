# factset_ec_scraper.py
# -*- coding: utf-8 -*-

import os, re, time
from datetime import datetime
from contextlib import contextmanager
from dateutil import parser as dateparser
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout, Page, Frame, Route, Request

# ========== 配置 ==========
START_URL = "https://my.apps.factset.com/workstation/calendar"
TICKER = "PERSONAL:NEW_TYPE3_MQ.OFDB"       # 可留空，手动切公司
OUTPUT_DIR = "output/Type_3"
HEADLESS = False                        # 建议 False，方便验证码/调试
WAIT_LOGIN_SECONDS = 150
ROW_LIMIT = None                        # 仅处理前 N 个；None 不限
DEBUG = True

# 翻页等待
NEXT_PAGE_EXTRA_WAIT = 30               # 原 20 → 10
NEXT_PAGE_MAX_WAIT   = 40               # 原 60 → 30
NEXT_PAGE_POLL_MS    = 300              # 原 400 → 300

# 文档页 / 抓取等待
DOC_NETWORKIDLE_TIMEOUT = 8000          # 原 25000 → 8000
DOC_SELECTOR_TIMEOUT     = 6000          # 等待 #sections 的最大时长
SCROLL_MAX_STEPS         = 12            # 原 40 → 12
SCROLL_WAIT_MS           = 80            # 原 120 → 80
MIN_SAVE_LEN             = 200           # 文本太短才二次抓取
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(*a):
    if DEBUG: print(*a)

def clean_filename(s: str) -> str:
    s = re.sub(r"[\\/*?\"<>|:]", "_", s)
    return re.sub(r"\s+", " ", s).strip()

def _norm_text(txt: str) -> str:
    return re.sub(r"\s+", " ", (txt or "").strip())

def _safe_stem(s: str) -> str:
    s = re.sub(r"[\\/*?\"<>|:]", "_", s)
    return re.sub(r"\s+", " ", s).strip()

def _parse_date_to_yyyy_mm_dd(s: str) -> str:
    """解析失败返回空串（不回退今天）。"""
    try:
        return dateparser.parse(_norm_text(s), fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return ""

def _date_from_onclick(a_loc) -> str:
    """优先从 onclick 的 openTranscript(..., 'yyyymmdd', ...) 提取日期。"""
    onclick = a_loc.get_attribute("onclick") or ""
    m_date = re.search(r'openTranscript\(\s*"\d+"\s*,\s*"[a-z]+"\s*,\s*"(\d{8})"', onclick, re.I)
    if not m_date:
        return ""
    ymd = m_date.group(1)
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"

def guess_date_from_text(text: str) -> str:
    m = re.search(r"\b(\d{1,2}[-/\s][A-Za-z\u4e00-\u9fa5]{3,}[-/\s]\d{4})\b", text) \
        or re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b", text)
    if m:
        try:
            return dateparser.parse(m.group(1), fuzzy=True).strftime("%Y-%m-%d")
        except Exception:
            return ""
    return ""

# ========== 资源拦截（核心提速点） ==========
def route_calendar(route: Route, request: Request):
    """列表/日历页：放行 document/script/xhr/css，拦截图片/媒体/字体。"""
    rtype = request.resource_type
    if rtype in ("image", "media", "font"):
        return route.abort()
    return route.continue_()

def route_document(route: Route, request: Request):
    """文档页：尽量只保留 document/script/xhr；拦截图片/媒体/字体/样式。"""
    rtype = request.resource_type
    if rtype in ("image", "media", "font", "stylesheet"):
        return route.abort()
    return route.continue_()

@contextmanager
def browser_ctx():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=HEADLESS, args=["--disable-features=PreloadMediaEngagementData,AutofillServerCommunication"])
        ctx = b.new_context(viewport={"width": 1400, "height": 900})
        ctx.set_default_timeout(7000)  # 降低默认超时
        page = ctx.new_page()

        # 针对两类页面分别设置路由
        ctx.route("**/workstation/calendar**", route_calendar)
        ctx.route("**/services/DocDisplay/**", route_document)
        ctx.route("**/document/**", route_document)
        yield page, ctx, b
        ctx.close(); b.close()

def wait_for_calendar(page: Page):
    page.goto(START_URL, wait_until="domcontentloaded")
    print("请完成登录与验证码，系统会进入 Calendar。")
    deadline = time.time() + WAIT_LOGIN_SECONDS
    markers = [
        "text=/Events\\s*&\\s*Transcripts/i",
        "text=/PAST\\s*EVENTS/i",
        "text=/Type\\s*Event/i",
        "text=/View All/i",
    ]
    while time.time() < deadline:
        try:
            if any(page.locator(m).first.is_visible() for m in markers):
                break
        except Exception:
            pass
        time.sleep(1.0)
    if DEBUG:
        page.screenshot(path=os.path.join(OUTPUT_DIR, "01_after_login.png"))

def maybe_jump_to_ticker(page: Page, ticker: str):
    if not ticker: return
    try:
        box = page.locator("input[placeholder*='Search FactSet'], input[type='search']").first
        if box.is_visible():
            box.click(); box.fill(ticker); box.press("Enter")
            page.wait_for_timeout(1200)
    except Exception:
        pass

# ---------- 选中承载网格的 iframe ----------
def pick_grid_frame(page: Page) -> Frame:
    scored = []
    for fr in page.frames:
        sc = 0
        for pat, w in [
            (r"/PAST\s*EVENTS/i", 3),
            (r"/Type\s*Event/i", 2),
            (r"/View All/i", 1),
        ]:
            try:
                if fr.locator(f"text={pat}").first.is_visible():
                    sc += w
            except Exception:
                pass
        scored.append((sc, fr))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1] if scored and scored[0][0] > 0 else page.main_frame
    return best

# ---------- 收集 Transcript 列（Indexed > Corrected > Raw） ----------
def collect_corrected_buttons(frame: Frame):
    results, seen = [], set()

    def _row_meta(a_loc):
        date_norm = _date_from_onclick(a_loc)
        row = a_loc.locator("xpath=ancestor::tr[1]")

        comp_cell = row.locator("td.compname, td[id$='_compname']").first
        comp_txt = _norm_text(comp_cell.inner_text()) if comp_cell.count() else ""

        q_cell = row.locator(
            "td[id$='_eventdeckphone'] .event_description a, "
            "td.daily_column_event .event_description a"
        ).first
        q_txt = _norm_text(q_cell.inner_text()) if q_cell.count() else ""
        m_q = re.search(r"\b(Q[1-4]\s+\d{4})\b", q_txt)
        quarter_short = m_q.group(1) if m_q else q_txt

        if not date_norm:
            date_cell = row.locator("td.datepart, td[id$='_datepart']").first
            raw = _norm_text(date_cell.inner_text()) if date_cell.count() else _norm_text(row.inner_text())
            date_norm = _parse_date_to_yyyy_mm_dd(raw) or guess_date_from_text(raw)
        return date_norm, comp_txt, quarter_short

    tds = frame.locator("td.daily_column_icon_link")
    count = tds.count()
    for i in range(count):
        td = tds.nth(i)
        links = td.locator("a")
        best = None
        best_rank = 10**9

        for j in range(links.count()):
            a = links.nth(j)
            alt   = (a.locator("img").first.get_attribute("alt") or "").lower()
            title = (a.get_attribute("title") or "").lower()
            aria  = (a.get_attribute("aria-label") or "").lower()
            label = " ".join([alt, title, aria])
            if "transcript" not in label:
                continue

            rank, variant = 30, "raw"
            if "indexed audio" in label:
                rank, variant = 1, "indexed"
            elif "corrected transcript" in label:
                rank, variant = 5, "corrected"
            elif "raw transcript" in label:
                rank, variant = 20, "raw"
            else:
                rank, variant = 25, "raw"

            if rank < best_rank:
                best_rank, best = rank, (a, variant)

        if not best:
            continue

        a, variant = best
        onclick = a.get_attribute("onclick") or ""
        m = re.search(r'openTranscript\(\s*"(\d+)"', onclick)
        tid = m.group(1) if m else None
        if not tid or tid in seen:
            continue
        seen.add(tid)

        date_norm, comp_txt, quarter_short = _row_meta(a)
        results.append({
            "id": tid, "date": date_norm, "company": comp_txt,
            "quarter": quarter_short, "variant": variant
        })

    if DEBUG:
        print(f"找到 Transcript 图标：{len(results)} 个（Indexed/Corrected/Raw）")
    return results

# ---------- 文档页抓正文（更快的等待 + 更快的滚动） ----------
def extract_transcript_from_viewer(doc_page: Page) -> str:
    try:
        doc_page.wait_for_load_state("domcontentloaded")
        # 尝试短 networkidle；失败也继续
        try:
            doc_page.wait_for_load_state("networkidle", timeout=DOC_NETWORKIDLE_TIMEOUT)
        except Exception:
            pass
    except Exception:
        pass

    outer_ifr_sel = (
        "iframe[title*='viewer-fusion-iframe'],"
        "iframe[data-testid='document-iframe'],"
        "iframe[data-qa-class='document-iframe'],"
        "iframe.doc-iframe"
    )
    outer_ifr = doc_page.locator(outer_ifr_sel).first
    try:
        outer_ifr.wait_for(state="visible", timeout=DOC_SELECTOR_TIMEOUT)
    except PwTimeout:
        return grab_sections_like_text(doc_page.main_frame)

    try:
        outer_fr = outer_ifr.element_handle().content_frame()
    except Exception:
        outer_fr = None

    if not outer_fr:
        return grab_sections_like_text(doc_page.main_frame)

    target_fr = None
    deadline = time.time() + 10  # 最多找 10s
    while time.time() < deadline and not target_fr:
        for fr in outer_fr.child_frames:
            try:
                if fr.locator("#sections, .sections").count() > 0:
                    target_fr = fr; break
                if "/services/DocDisplay/getstory" in (fr.url or ""):
                    target_fr = fr; break
            except Exception:
                continue
        if not target_fr:
            try: outer_fr.wait_for_timeout(200)
            except Exception: pass

    if not target_fr:
        for fr in doc_page.frames:
            try:
                if fr.locator("#sections, .sections").count() > 0:
                    target_fr = fr; break
            except Exception:
                continue
    if not target_fr:
        target_fr = outer_fr

    try:
        target_fr.wait_for_selector("#sections, .sections", timeout=DOC_SELECTOR_TIMEOUT)
    except Exception:
        pass

    return grab_sections_like_text(target_fr)

def extract_participants_text(fr: Frame) -> str:
    try:
        root = fr.locator("#participants, .participants").first
        if not root.count():
            return ""
        parts = []
        try:
            hdr = root.locator(".participantHeader").first
            if hdr.count():
                t = hdr.inner_text().strip()
                if t: parts.append(t)
        except Exception:
            pass
        try:
            subhdrs = root.locator(".ptHeader")
            for i in range(subhdrs.count()):
                t = subhdrs.nth(i).inner_text().strip()
                if t: parts.append(t)
        except Exception:
            pass
        try:
            rows = root.locator(".ptSmallCell")
            for i in range(rows.count()):
                row = rows.nth(i)
                name = ""
                role = ""
                try: name = row.locator(".ptName").first.inner_text().strip()
                except Exception: pass
                try: role = row.locator(".ptInfo").first.inner_text().strip()
                except Exception: pass
                line = " — ".join([v for v in [name, role] if v])
                if line: parts.append(line)
        except Exception:
            pass
        if not parts:
            txt = root.inner_text().strip()
            if txt: parts.append(txt)
        return "\n".join(parts)
    except Exception:
        return ""

def grab_sections_like_text(fr: Frame) -> str:
    # 自适应滚动触发懒加载
    try:
        last_h = 0
        for _ in range(SCROLL_MAX_STEPS):
            fr.evaluate("window.scrollBy(0, 2400)")
            fr.wait_for_timeout(SCROLL_WAIT_MS)
            h = fr.evaluate("() => document.scrollingElement ? document.scrollingElement.scrollHeight : document.body.scrollHeight")
            if abs(h - last_h) < 5:
                break
            last_h = h
    except Exception:
        pass

    output_blocks = []

    # Participants（可选开销很小）
    ptxt = extract_participants_text(fr)
    if ptxt: output_blocks.append(ptxt)

    # Sections（结构化抓取）
    try:
        if fr.locator("#sections, .sections").count() > 0:
            headers = fr.locator("#sections .sectionHeader, .sections .sectionHeader")
            paras   = fr.locator("#sections .pwrapper, .sections .pwrapper")
            parts = []
            for i in range(headers.count()):
                t = headers.nth(i).inner_text().strip()
                if t: parts.append(t)
            for i in range(paras.count()):
                t = paras.nth(i).inner_text().strip()
                if t: parts.append(t)
            if not parts:
                cont = fr.locator("#sections, .sections").first
                if cont.count():
                    txt = cont.inner_text().strip()
                    if txt: parts.append(txt)
            if parts:
                output_blocks.append("\n\n".join(parts))
    except Exception:
        pass

    if not output_blocks:
        try:
            body_txt = (fr.locator("body").inner_text() or "").strip()
            if body_txt: output_blocks.append(body_txt)
        except Exception:
            pass
    return "\n\n".join(output_blocks)

# ---------- Next Page ----------
def find_next_link(frame: Frame):
    sel = (
        "#navigation a:has-text('Next Page'), "
        "a:has-text('Next Page'), "
        "a[href*=\"goToPage('50', 'next')\" i], "
        "a[href*='goToPage'][href*=\"'next'\" i]"
    )
    link = frame.locator(sel).first
    if link.count() > 0 and link.is_visible():
        return link
    return None

def goto_next_page(page: Page) -> bool:
    fr = pick_grid_frame(page)
    next_link = find_next_link(fr)
    if not next_link:
        return False

    def _first_row_signature(f: Frame) -> str:
        try:
            row = f.locator("tr").first
            if row.count():
                date_txt = _norm_text(row.locator("td.datepart, td[id$='_datepart']").first.inner_text())
                comp_txt = _norm_text(row.locator("td.compname, td[id$='_compname']").first.inner_text())
                return f"{date_txt}|{comp_txt}"
        except Exception:
            pass
        return ""

    old_start = None
    try:
        rs = fr.locator("#navigation .record_start, .record_start").first
        if rs.count():
            old_start = (rs.inner_text() or "").strip()
    except Exception:
        pass
    old_sig = _first_row_signature(fr)

    next_link.click()

    loops = int((NEXT_PAGE_MAX_WAIT * 1000) / NEXT_PAGE_POLL_MS)
    for _ in range(max(loops, 1)):
        try:
            fr.wait_for_timeout(NEXT_PAGE_POLL_MS)
            fr = pick_grid_frame(page)
            rs = fr.locator("#navigation .record_start, .record_start").first
            if rs.count():
                cur = (rs.inner_text() or "").strip()
                if old_start and cur != old_start and cur != "":
                    if NEXT_PAGE_EXTRA_WAIT > 0:
                        fr.wait_for_timeout(NEXT_PAGE_EXTRA_WAIT * 1000)
                    return True
            if old_sig:
                if _first_row_signature(fr) not in ("", old_sig):
                    if NEXT_PAGE_EXTRA_WAIT > 0:
                        fr.wait_for_timeout(NEXT_PAGE_EXTRA_WAIT * 1000)
                    return True
            nl = find_next_link(fr)
            if not nl or (nl.count() and not nl.is_enabled()):
                if NEXT_PAGE_EXTRA_WAIT > 0:
                    fr.wait_for_timeout(NEXT_PAGE_EXTRA_WAIT * 1000)
                return True
        except Exception:
            pass

    if NEXT_PAGE_EXTRA_WAIT > 0:
        try: fr.wait_for_timeout(NEXT_PAGE_EXTRA_WAIT * 1000)
        except Exception: pass
    return True

# ---------- 点击（现取现点 + 重试） ----------
def robust_click_transcript(page: Page, transcript_id: str) -> Page | None:
    for attempt in range(3):
        fr = pick_grid_frame(page)
        try: fr.evaluate("window.scrollBy(0, 200)")
        except Exception: pass

        a = None
        if transcript_id:
            a = fr.locator(f"a[data-qa-id='open-transcript-{transcript_id}']").first
        if not a or not a.count():
            a = fr.locator("td.daily_column_icon_link a:has(img[alt*='Transcript' i])").first

        if not a or not a.count():
            fr.wait_for_timeout(250)
            continue

        try:
            a.scroll_into_view_if_needed(timeout=1200)
        except Exception:
            pass

        try:
            with page.context.expect_page(timeout=3500) as ev:
                a.click()
            return ev.value
        except PwTimeout:
            pass

        try:
            a.click(timeout=3500)
            return page
        except Exception:
            fr.wait_for_timeout(250)
            continue
    return None

# ---------- 主流程 ----------
def run():
    with browser_ctx() as (page, ctx, b):
        wait_for_calendar(page)
        maybe_jump_to_ticker(page, TICKER)

        processed_total = 0
        page_idx = 1
        stop_due_to_row_limit = False

        while True:
            frame = pick_grid_frame(page)
            buttons = collect_corrected_buttons(frame)
            print(f"[第 {page_idx} 页] 本页找到 Transcript 图标：{len(buttons)} 个（Indexed/Corrected/Raw）")

            for idx, item in enumerate(buttons, start=1):
                if ROW_LIMIT is not None and processed_total >= ROW_LIMIT:
                    stop_due_to_row_limit = True
                    break

                tid      = item.get("id")
                date_str = item.get("date", "")
                company  = item.get("company") or (TICKER or "FACTSET")
                quarter  = item.get("quarter", "")
                variant  = item.get("variant", "")

                print(f"  [{idx}/{len(buttons)} @第{page_idx}页] {date_str or '?'} - {company} - {quarter} ({variant}) - 打开文档…")

                newp = robust_click_transcript(page, tid)
                if not newp:
                    print("    × 点击失败，跳过。")
                    continue

                try:
                    newp.wait_for_load_state("domcontentloaded", timeout=6000)
                except PwTimeout:
                    print("    × 文档页加载超时，跳过。")
                    if newp is not page: newp.close()
                    continue

                try:
                    text = extract_transcript_from_viewer(newp)
                except Exception as e:
                    print("    × 提取正文失败：", e)
                    if newp is not page: newp.close()
                    continue

                if text and len(text) < MIN_SAVE_LEN:
                    # 只在很短时做二次尝试
                    try:
                        text = extract_transcript_from_viewer(newp)
                    except Exception:
                        pass

                date_for_name = date_str or datetime.today().strftime("%Y-%m-%d")
                if not date_str:
                    print("    ！该行未解析出日期，使用当前日期作为文件名的一部分（fallback）。")
                parts = [_safe_stem(company), date_for_name]
                if quarter:
                    parts.append(_safe_stem(quarter))
                fname = clean_filename("_".join(parts) + ".txt")

                outp = os.path.join(OUTPUT_DIR, fname)
                with open(outp, "w", encoding="utf-8") as f:
                    f.write(text or "")
                print(f"    ✓ 已保存：{outp}（{len(text or '')} 字符）")

                processed_total += 1

                if newp is not page:
                    newp.close()

                # 处理完一条后，frame 可能刷新，重新抓一次（避免句柄失效）
                frame = pick_grid_frame(page)

            if stop_due_to_row_limit:
                break

            if not goto_next_page(page):
                break
            page_idx += 1

        print(f"完成。共导出 {processed_total} 个 txt 到 ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n用户中断。")
