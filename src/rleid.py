import copy


def rleid(a):
    a = list(a)
    if len(set(a)) < 1:
        return a
    else:
        a_copy = copy.copy(a)
        a_copy.append('tail')
        counter = 0
        class_symbol = 1
        final_list = []

        for i in range(0, len(a)):
            counter += 1

            if a_copy[i] != a_copy[i + 1]:
                sub_seq_length = counter
                counter = 0
                final_list.append([class_symbol] * sub_seq_length)
                class_symbol += 1
            output = [y for x in final_list for y in x]

        return output
