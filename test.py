
def make_visual_evaluation(query):

    rquery = ""
    if "&" in query or "|" in query or "~" in query:
        rquery = query
    else:
        splited = query.split()
        for w in range(len(splited)):
            if w == len(splited) - 1:
                rquery = rquery + splited[w]
                break
            rquery = rquery + splited[w] + " " + "&" + " "
    print(rquery)


[print(q) for q in [1,2,3,2]]
q = "Hola adios sennor feo"
make_visual_evaluation(q)