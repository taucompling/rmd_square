import numpy as np
def check_literal_or_pragmatic(type, lexica, puzzle):
    if type > len(lexica)-1 or puzzle:
        return "pragmatic"
    return "literal"

def get_lexica_representations(type, lexica, puzzle):
    l = lexica[type-len(lexica)]
    for column in l.T:
        if -5 in column:
            l = np.delete(l, np.where(l == -5)[1][0], 1)
    return f"{check_literal_or_pragmatic(type, lexica, puzzle)}:\n{l}"

def find_best_x_prag_lit(results, lexica, x, puzzle):
    """finds the x best pragmatic and literal results

    :param results: proportion of last epoch
    :type results: list
    :param lexica: lexica of types
    :type lexica: np.array
    :param x: desired number of best types
    :type x: int
    :return: two lists, one for best x literal types and of for best x pragmatic types
    :rtype: lists
    """
    sorted_idx = sorted(range(len(results)), key=lambda k: results[k], reverse=True) # list with types sorted by highest prob 
    x_best_pragmatic, x_best_literal = [], []
    for type in sorted_idx:
        if puzzle: 
            if len(x_best_pragmatic) == x:
                return x_best_literal, x_best_pragmatic

        if len(x_best_literal) == x and len(x_best_pragmatic) == x:
            return x_best_literal, x_best_pragmatic

        elif check_literal_or_pragmatic(type, lexica, puzzle) == "literal" and len(x_best_literal) <= x:
            x_best_literal.append(type)
        elif check_literal_or_pragmatic(type, lexica, puzzle) == "pragmatic" and len(x_best_pragmatic) <= x:
            x_best_pragmatic.append(type)

    raise Exception("Something went wrong!")
    

def print_best_x_types_to_file(results, lexica, result_path, x, puzzle):
    x_best_literal, x_best_pragmatic = find_best_x_prag_lit(results, lexica, x, puzzle)
    list_to_print = []
    for type in x_best_literal + x_best_pragmatic:
        list_to_print.append(f"-Type {type} with proportion {results[type]}\n")
        list_to_print.append(f"{(get_lexica_representations(type, lexica, puzzle))}\n\n")

    with open(f"{result_path}/results/lexica_best_{x}_lit_and_prag_types.txt", "w") as file:
        gap = "literal and " if not puzzle else ""
        file.write(f"Best {x} type(s) and their proportion for {gap}pragmatic types:\n\n")
        file.writelines(list_to_print)