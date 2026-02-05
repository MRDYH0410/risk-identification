import os  # 处理路径/目录
import json  # 导出 JSON
import argparse  # 命令行参数
import pandas as pd  # 读 CSV/处理表
from typing import Dict, List  # 类型注解


def script_dir() -> str:
    """返回本脚本所在目录的绝对路径（与 PyCharm 运行位置无关）。"""
    return os.path.dirname(os.path.abspath(__file__))


def auto_find_inputs() -> List[str]:
    """
    在脚本目录下的 ./standardized_lexicons/ 自动查找 *_standardized.csv。
    找不到则返回空列表。
    """
    base = os.path.join(script_dir(), "standardized_lexicons")             # 拼目录：<脚本目录>/standardized_lexicons
    if not os.path.isdir(base):                                            # 目录不存在就返回空
        return []
    files = []
    for fn in os.listdir(base):                                            # 遍历目录文件
        if fn.lower().endswith("_standardized.csv"):                       # 只要以 _standardized.csv 结尾的
            files.append(os.path.join(base, fn))                           # 记录绝对路径
    return sorted(files)                                                   # 排序后返回


def load_standardized(path: str) -> pd.DataFrame:
    """
    加载单个标准化 CSV（bucket, term, source, description）。
    做基本字段校验、空值清理和去重。
    """
    df = pd.read_csv(path)                                                 # 读取 CSV
    need_cols = {"bucket", "term"}                                         # 强制要求这两列
    if not need_cols.issubset(set(df.columns)):                            # 缺列就报错
        raise ValueError(f"{path} 缺少列：{need_cols - set(df.columns)}")
    df["bucket"] = df["bucket"].astype(str).str.strip()                    # 去空格
    df["term"] = df["term"].astype(str).str.strip()                        # 去空格
    df = df[(df["bucket"] != "") & (df["term"] != "")].drop_duplicates()   # 去空/去重
    return df


def build_risk_lexicons(std_files: List[str]) -> Dict[str, List[str]]:
    """
    合并多个标准化 CSV 为 dict：{bucket: [term1, term2, ...]}。
    """
    all_df = []                                                            # 收集 DataFrame
    for p in std_files:                                                    # 遍历输入文件
        if not os.path.exists(p):                                          # 不存在就报错
            raise FileNotFoundError(f"File not found: {p}")
        all_df.append(load_standardized(p))                                # 读取并加入
    if not all_df:                                                         # 一个都没有
        return {}                                                          # 返回空 dict

    df = pd.concat(all_df, ignore_index=True)                              # 纵向合并
    out: Dict[str, List[str]] = {}                                         # 输出 dict
    for bkt, grp in df.groupby("bucket"):                                  # 按 bucket 分组
        terms = sorted(set(grp["term"].tolist()))                          # 去重并排序
        out[bkt] = terms                                                   # 写入 dict
    return out


def main():
    """主入口：解析参数 -> 自动发现输入(可选) -> 合并 -> 打印/导出 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", default=None,                               # 允许为空，留给自动扫描补上
        help="One or more standardized CSVs (from lexicon_standardizer.py)"
    )
    parser.add_argument(
        "--out_json", type=str, default=None,                              # 可选的 JSON 输出路径
        help="Optional path to save merged dict as JSON"
    )
    args = parser.parse_args()                                             # 解析参数

    # 如果用户没有传 --inputs，就自动扫描 ./standardized_lexicons/
    std_files = args.inputs if args.inputs else auto_find_inputs()

    # 扫描不到任何输入时，给出非常明确的报错指引（包含当前工作目录 & 期望路径）
    if not std_files:
        example1 = os.path.join(script_dir(), "standardized_lexicons", "lm_standardized.csv")
        example2 = os.path.join(script_dir(), "standardized_lexicons", "flr_standardized.csv")
        raise SystemExit(
            "未找到任何标准化词典 CSV。\n"
            f"请确认存在目录：{os.path.dirname(example1)}\n"
            f"示例文件：\n  - {example1}\n  - {example2}\n"
            "或者请通过参数显式指定：\n"
            "  python build_risk_lexicons.py --inputs <path1> <path2> ... [--out_json out.json]"
        )

    print("将合并以下文件：")
    for p in std_files:
        print("  -", p)

    lexicons = build_risk_lexicons(std_files)                              # 合并为 dict
    print(f"Buckets 总数：{len(lexicons)}")
    for bkt, terms in lexicons.items():
        print(f"  · {bkt}: {len(terms)} terms")

    # 如果指定了 JSON 输出，就保存到文件
    if args.out_json:
        out_path = args.out_json                                           # 直接使用用户路径
    else:
        out_path = os.path.join(script_dir(), "risk_lexicons.json")        # 默认导出到脚本目录
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lexicons, f, ensure_ascii=False, indent=2)
    print(f"已保存 JSON -> {out_path}")


if __name__ == "__main__":
    main()