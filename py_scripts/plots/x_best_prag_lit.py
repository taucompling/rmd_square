def check_literal_or_pragmatic(type, lexica):
    if type > len(lexica)-1:
        return "pragmatic"
    else:
        return "literal"

def get_lexica_representations(type, lexica):
    return f"{check_literal_or_pragmatic(type, lexica)}:\n{lexica[type-len(lexica)]}"

def find_best_x_prag_lit(results, lexica, x):
    """finds the x best pragmatic and literal results

    :param results: proportion of last epoch
    :type results: list
    :param lexica: lexica of types
    :type lexica: np.arry
    :param x: desired number of best types
    :type x: int
    :return: two lists, one for best x literal types and of for best x pragmatic types
    :rtype: lists
    """
    sorted_idx = sorted(range(len(results)), key=lambda k: results[k], reverse=True) # list with types sorted by highest prob 
    six_best_pragmatic, six_best_literal = [], []
    for type in sorted_idx:
        if len(six_best_literal) == 6 and len(six_best_pragmatic) == x:
            break
        elif check_literal_or_pragmatic(type, lexica) == "literal" and len(six_best_literal) <= x:
            six_best_literal.append(type)
        elif check_literal_or_pragmatic(type, lexica) == "pragmatic" and len(six_best_pragmatic) <= x:
            six_best_pragmatic.append(type)

    return six_best_literal, six_best_pragmatic

def print_best_x_types_to_file(results, lexica, result_path, x):
    x_best_literal, x_best_pragmatic = find_best_x_prag_lit(results, lexica, x)
    list_to_print = []
    for type in x_best_literal + x_best_pragmatic:
        list_to_print.append(f"-Type {type} with proportion {results[type]}\n")
        list_to_print.append(f"{(get_lexica_representations(type, lexica))}\n\n")

    with open(f"experiments/{result_path}/results/lexica_best_{x}_lit_and_prag_types.txt", "w") as file:
        file.writelines(list_to_print)