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

    right_eyebrow = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
    left_eyebrow =  [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

    for i in range(len(right_eyebrow)):
        E.append((right_eyebrow[i], right_eyebrow[(i+1) % len(right_eyebrow)]))
        E.append((6, right_eyebrow[i]))

    for i in range(len(left_eyebrow)):
        E.append((left_eyebrow[i], left_eyebrow[(i+1) % len(left_eyebrow)]))
        E.append((6, left_eyebrow[i]))

    pupils = [468, 473]

    for i in range(len(pupils)):
        E.append((6, pupils[i]))
        E.append((4,pupils[i]))

    for i in pupils + [6,4]:
        E.append((10, i))

    Tmp = E.copy()
    for i in Tmp:
        E.append((i[1], i[0]))

    V = V + right_eyebrow + left_eyebrow + pupils + [6] + [10]
    
    return V, E


def bi_graph_norm():
    #    0, 1,  2  , 3  , 4  , 5  , 6 ,  7 ,  8 ,  9 , 10,  11, 12,  13 ,14,  15 ,16, 17,  18, 19, 20
    V = [4, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
    right_eyebrow = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
    left_eyebrow =  [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    pupils = [468, 473]
    gaze = [10]

    temp = V + right_eyebrow + left_eyebrow + pupils + [6] + gaze
    num = [i for i in range(len(temp))]
    D = dict(zip(temp, num)) # dictionary of all the nodes
    
    E = []

    # connect mouth
    for i in range(len(V) - 2):
        E.append((D[V[i + 1]], D[V[i + 2]]))
    E.append((D[V[-1]], D[V[1]]))  # close mouth circle

    # connect to center node
    for i in range(len(V) - 1):
        E.append((D[V[0]], D[V[i + 1]]))

    # connect lower lip and upper lip
    E.append((D[39], D[181]))
    E.append((D[37], D[84]))
    E.append((D[0], D[17]))
    E.append((D[267], D[314]))
    E.append((D[269], D[405]))

    for i in range(len(right_eyebrow)):
        E.append((D[right_eyebrow[i]], D[right_eyebrow[(i+1) % len(right_eyebrow)]]))
        E.append((D[6], D[right_eyebrow[i]]))

    for i in range(len(left_eyebrow)):
        E.append((D[left_eyebrow[i]], D[left_eyebrow[(i+1) % len(left_eyebrow)]]))
        E.append((D[6], D[left_eyebrow[i]]))

    pupils = [468, 473]

    for i in range(len(pupils)):
        E.append((D[6], D[pupils[i]]))
        E.append((D[4], D[pupils[i]]))

    for i in pupils + [6,4]:
        E.append((D[10], D[i]))

    Tmp = E.copy()
    for i in Tmp:
        E.append((i[1], i[0]))
    
    V = num
    return V, E