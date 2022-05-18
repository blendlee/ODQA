queue1 = [3, 2, 7, 2]
queue2 = [4, 6, 5, 1]
    
que=queue1+queue2
start=0
end=0
target=sum(que)//2
if sum(que)%2==1:
    print(-1)


result=-1
while start <=end or end!=len(que):
    if start==end:
        if que[start]==target:
            result=(start,end)
            break
        else:
            end+=1
    else:
        if sum(que[start:end+1]) < target:
            end+=1
        elif sum(que[start:end+1]) > target:
            start+=1
        else:
            result=(start,end)
            break
if result==-1:
    print(-1)
start,end = result
print(1)
if start==end:
    pass
else:
    print(start+end-len(queue1))