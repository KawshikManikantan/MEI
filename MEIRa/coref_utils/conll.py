import re
import subprocess
import operator
import collections

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) "
    r"([0-9.]+)%\tF1: ([0-9.]+)%.*",
    re.DOTALL,
)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def output_conll(input_file, output_file, predictions, subtoken_map):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [
                cluster_id
                for cluster_id, end in sorted(
                    v, key=operator.itemgetter(1), reverse=True
                )
            ]
        for k, v in end_map.items():
            end_map[k] = [
                cluster_id
                for cluster_id, start in sorted(
                    v, key=operator.itemgetter(1), reverse=True
                )
            ]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            # output_file.write("\n")
        else:
            assert get_doc_key(row[0], row[1]) == doc_key
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1


def official_conll_eval(
    conll_scorer, gold_path, predicted_path, metric, official_stdout=False
):
    cmd = [conll_scorer, metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print(stderr)

    if official_stdout:
        print("Official result for {}".format(metric))
        print(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(
    conll_scorer,
    gold_path,
    predictions,
    subtoken_maps,
    prediction_path,
    all_metrics=False,
    official_stdout=False,
):
    with open(prediction_path, "w") as prediction_file:
        with open(gold_path, "r") as gold_file:
            output_conll(gold_file, prediction_file, predictions, subtoken_maps)

    result = {
        metric: official_conll_eval(
            conll_scorer, gold_file.name, prediction_file.name, metric, official_stdout
        )
        for metric in ("muc", "bcub", "ceafe")
    }
    return result
