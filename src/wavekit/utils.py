import re
from typing import Any


# just like shell brace expansion
# or regex pattern
def expand_pattern(pattern: str) -> dict[tuple, str]:
    # 数字范围模式
    range_pattern = re.compile(r"\{(\d+)\.\.(\d+)(?:\.\.(\d+))?\}")
    # 字符串列表模式
    list_pattern = re.compile(r"\{([^\{\}]+)\}")
    # 正则表达式模式
    regex_pattern = re.compile(r"<(.+)>")

    # 处理数字范围模式
    def expand_range(match):
        start = int(match.group(1))
        end = int(match.group(2))
        step = int(match.group(3)) if match.group(3) else 1
        return [x for x in range(start, end + 1, step)]

    # 处理字符串列表模式
    def expand_list(match):
        return match.group(1).split(",")

    def expand_regex(match):
        return f"({match.group(1)})"

    # 递归解析模式
    def recursive_expand(parts):
        if not parts:
            return {(): ""}

        first_part = parts[0]
        rest_parts = parts[1:]

        expanded_first = []

        if match := range_pattern.match(first_part):
            expanded_first = {(p,): str(p) for p in expand_range(match)}
        elif match := list_pattern.match(first_part):
            expanded_first = {(p,): p for p in expand_list(match)}
        elif match := regex_pattern.match(first_part):
            expanded_first = {(None,): expand_regex(match)}
        else:
            expanded_first = {(): first_part}

        expanded_rest = recursive_expand(rest_parts)

        return {
            fp + rp: fs + rs
            for fp, fs in expanded_first.items()
            for rp, rs in expanded_rest.items()
        }

    # 分割输入模式
    parts = []
    i = 0
    expand_pattern_pairs = {"{": "}", "<": ">"}
    print(pattern)
    while i < len(pattern):
        if pattern[i] in expand_pattern_pairs:
            print(">>>>>>>>>>>>",expand_pattern_pairs[pattern[i]], pattern[i:])
            j = pattern.find(expand_pattern_pairs[pattern[i]], i)
            if j == -1:
                raise ValueError("Unmatched pattern delimiter: " + pattern[i])
            parts.append(pattern[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < len(pattern) and pattern[j] not in expand_pattern_pairs:
                j += 1
            parts.append(pattern[i:j].replace("[", r"\[").replace("]", r"\]"))
            i = j

    return recursive_expand(parts)
