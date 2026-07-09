import pandas as pd
import sys
import json


def export_to_json(stats_dict):
    """Export the stats dict to JSON"""
    export_data = {
        "features": list(stats_dict.keys()),
        "mean": {feat: stats_dict[feat]["Mean"] for feat in stats_dict},
        "std": {feat: stats_dict[feat]["Std"] for feat in stats_dict}
    }

    with open("model.json", "w") as f:
        json.dump(export_data, f, indent=4)


def print_stats_table(stats_dict, stat_names):
    """ Prints the table displaying the stats of each feature"""
    features = list(stats_dict.keys())

    MAX_WIDTH = 14

    col_widths = {}
    for feat in features:
        col_widths[feat] = min(max(len(feat), 12), MAX_WIDTH) + 1

    header = f"{'':>10}"
    for feat in features:
        if len(feat) > MAX_WIDTH:
            display_name = feat[:MAX_WIDTH-2] + ".."
        else:
            display_name = feat

        header += f" {display_name:>{col_widths[feat]}}"
    print(header)

    for stat in stat_names:
        row_str = f"{stat:>10}"

        for feat in features:
            val = stats_dict[feat][stat]
            val_str = f"{val:.6f}"

            row_str += f" {val_str:>{col_widths[feat]}}"

        print(row_str)


def mean(nums, size):
    """Calculate the Mean of a list of ints"""
    total = 0
    for n in nums:
        total += n
    return float(total / size)


def median(nums, size):
    """Calculate the Median of a list of ints"""
    sorted_nums = sorted(nums)
    mid = size // 2

    if size % 2 == 0:
        return float((sorted_nums[mid - 1] + sorted_nums[mid]) / 2)
    else:
        return sorted_nums[mid]


def quartile(nums, size):
    """Calculate the Quartiles of a list of ints"""
    sorted_nums = sorted(nums)

    q1_index = size // 4
    q2_index = (size * 3) // 4

    q1 = float(sorted_nums[q1_index])
    q2 = float(sorted_nums[q2_index])

    return [q1, q2]


def var(nums, size):
    """Calculates the Variance of a list of ints"""
    m = mean(nums, size)

    sum_squared_diff = 0
    for n in nums:
        difference = n - m
        sum_squared_diff += difference ** 2

    return float(sum_squared_diff / size)


def std(nums, size):
    """Calculates the Standard Deviation of a list of ints"""
    return var(nums, size) ** 0.5


def find_min(nums):
    """Returns the smallest number in a list"""
    min_n = nums[0]
    for n in nums:
        if n < min_n:
            min_n = n
    return min_n


def find_max(nums):
    """Returns the biggest number in a list"""
    max_n = nums[0]
    for n in nums:
        if n > max_n:
            max_n = n
    return max_n


def main():
    """
        Prints the following stats for each feature in the
        csv passed as an argument:
        Count, Mean, Std, Min, q1, median, q3, Max
        With -bonus: Var, Range, IQR, CV
    """

    csv_file = sys.argv[1]
    bonus = (len(sys.argv) == 3 and sys.argv[2] == '-bonus')

    df = pd.read_csv(csv_file)
    if 'Index' in df.columns:
        df = df.drop('Index', axis='columns')
    numeric_df = df.select_dtypes(include=['number'])
    features = numeric_df.columns

    stats_dict = {}

    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    if bonus:
        stat_names.extend(['Var', 'Range', 'IQR', 'CV'])

    for feature in features:
        data = numeric_df[feature].dropna().tolist()
        count = len(data)

        if count == 0:
            continue

        mean_val = mean(data, count)
        std_val = std(data, count)
        min_val = find_min(data)
        max_val = find_max(data)
        q1, q3 = quartile(data, count)
        median_val = median(data, count)

        feature_stats = {
            "Count": count,
            "Mean": mean_val,
            "Std": std_val,
            "Min": min_val,
            "25%": q1,
            "50%": median_val,
            "75%": q3,
            "Max": max_val
        }

        if bonus:
            feature_stats["Var"] = var(data, count)
            feature_stats["Range"] = float(max_val - min_val)
            feature_stats["IQR"] = float(q3 - q1)
            if mean_val != 0:
                feature_stats["CV"] = float(std_val / mean_val)
            else:
                feature_stats["CV"] = 0.0

        stats_dict[feature] = feature_stats

    export_to_json(stats_dict)
    print_stats_table(stats_dict, stat_names)


if __name__ == '__main__':
    try:
        usage = "Usage: describe.py <file.csv> [-bonus]"
        assert len(sys.argv) in [2, 3], f"Invalid number of arguments\n{usage}"
        if len(sys.argv) == 3:
            assert sys.argv[2] == '-bonus', f"Invalid argument\n{usage}"
        main()
    except Exception as e:
        print(e)
