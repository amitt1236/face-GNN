def bi_graph_norm():
    #    0, 1,  2  , 3  , 4  , 5  , 6 ,  7 ,  8 ,  9 , 10,  11, 12,  13 ,14,  15 ,16, 17,  18, 19, 20
    V = [4, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
    E = []

    # connect mouth
    for i in range(len(V) - 2):
        E.append((i + 1, i + 2))

    E.append((20, 1))  # close mouth circle

    # connect to center node
    for i in range(len(V) - 1):
        E.append((0, i + 1))

    # connect lower lip and upper lip
    E.append((19, 13))  # (39,181)
    E.append((20, 12))  # (37,84)
    E.append((1, 11))  # (0,17)
    E.append((2, 10))  # (267,314)
    E.append((3, 9))  # (269,405)

    Tmp = E.copy()

    for i in Tmp:
        E.append((i[1], i[0]))

    return V, E


def bi_graph():
    V = [4, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
    E = []

    # connect mouth
    for i in range(len(V) - 2):
        E.append((V[i + 1], V[i + 2]))
    E.append((V[-1], V[1]))  # close mouth circle

    # connect to center node
    for i in range(len(V) - 1):
        E.append((V[0], V[i + 1]))

    # connect lower lip and upper lip
    E.append((39, 181))
    E.append((37, 84))
    E.append((0, 17))
    E.append((267, 314))
    E.append((269, 405))

    Tmp = E.copy()

    for i in Tmp:
        E.append((i[1], i[0]))

    return V, E