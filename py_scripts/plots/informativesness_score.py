import numpy as np

def get_type_bin(type_idx, bins):

    for b in range(len(bins)):
        if type_idx in bins[b]:
            return b
    raise Exception("Something did not work out with the bins!")

def get_informativeness(pPrime, typeList, bins, result_path):

    
    inventories = { ((1,1,0,0), (0,1,1,0), (0,1,1,0), (0,0,1,1)) : ["AND, OR, NAND, NOR", []],
                    ((1,0,0), (0,1,0), (0,1,0), (0,1,1)) : ["AND, NAND, NOR", []],
                    ((1,0,0), (1,1,0), (1,1,0), (0,1,1)) : ["OR, NAND, NOR", []],
                    ((1,1,0), (0,1,0), (0,1,0), (0,0,1)) : ["AND, OR, NOR ", []],
                    ((1,1,0), (0,1,1), (0,1,1), (0,0,1)) : ["AND, OR, NAND", []],
                    ((0,0), (1,0), (1,0), (1,1)) : ["NAND, NOR", []],
                    ((1,0), (0,0), (0,0), (0,1)) : ["AND, NOR", []],
                    ((1,0), (1,0), (1,0), (0,1)) : ["OR, NOR", []],
                    ((1,0), (0,1), (0,1), (0,1)) : ["AND,NAND", []],
                    ((1,0), (1,1), (1,1), (0,1)) : ["OR, NAND", []],
                    ((1,1), (0,1), (0,1), (0,0)) : ["AND, OR", []],
                    ((0), (0), (0), (1)) : ["NOR", []],
                    ((0), (1), (1), (1)) : ["NAND", []],
                    ((1), (0), (0), (0)) : ["AND", []],
                    ((1), (1), (1), (0)) : ["OR", []]}
  
    bin_inventories = dict() # dictionary with bins instead of lexica

    for lex, descr in inventories.items():
        lex_id = [index for index, typ in enumerate(typeList) if np.array_equal(typ.lexicon, np.array(lex))]
        if lex_id:
            lex_bin = tuple(bins[get_type_bin(lex_id[0], bins)])
            bin_inventories[lex_bin] = descr


    with open(result_path + "/results/informativeness_score.txt", "w+") as output_file:
        output_file.write("Informativeness Scores for the following inventories: \n\n")
        for lex_bin, descr in bin_inventories.items():
            for i, typ in enumerate(typeList):
                l = typ.lexicon

                for column in l.T:
                    if -5 in column:
                        l = np.delete(l, np.where(l == -5)[1][0], 1)

                if i in lex_bin:
                    descr[1].append(pPrime[i])


            output_file.write(f"Inventory {descr[0]}:\nAverage Informativity: {round(np.sum(descr[1]), 2)}, Maximum Informativity: {round(max(descr[1]), 2)}\n\n")


