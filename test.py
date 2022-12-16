
def make_visual_evaluation(query):

    rcuery = ""
    if "&" in query or "|" in query or "~" in query:
            rcuery = query
    else:
        splited = query.split()
        for w in range(len(splited)):
            if w == len(splited) - 1:
                rcuery = rcuery + splited[w]
                break
            rcuery = rcuery + splited[w] + " " + "&" + " "
    print(rcuery)


q = "Hola  & adios sennor feo"
make_visual_evaluation(q)