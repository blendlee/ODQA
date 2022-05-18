from collections import deque

rc = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
operations=	["Rotate", "ShiftRow"]
r=len(rc)
c=len(rc[0])

def Rotate(graph,r,c):
    edges=graph[0]+[graph[i][-1] for i in range(1,c-1)]+list(reversed(graph[-1]))+list(reversed([graph[i][0] for i in range(1,c-1)]))
    edges=deque(edges)
    print(edges)
    edges.rotate(1)
    edges=list(edges)
    mid=len(edges)//2
    graph[0] = edges[:r]
    graph[-1] = list(reversed(edges[mid:mid+r]))
    for i in range(1,c-1):
        graph[i][-1]=edges[r+i-1]
        graph[i][0]=edges[-i]
    
    return graph

def shiftrow(graph):
    new_graph=deque(graph)
    new_graph.rotate(1)
    return list(new_graph)

rc=shiftrow(rc)
rc= Rotate(rc,r,c)
print(rc)