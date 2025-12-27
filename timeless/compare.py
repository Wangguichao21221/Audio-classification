dif = 0
with open("./submit.csv") as f1:
  with open("./submission_vote.csv") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    for i in range(len(lines1)):
      if lines1[i] != lines2[i]:
        print(f"Different at line {i+1}, \n{lines1[i]}{lines2[i]}")
        dif+=1
print(f'{dif} in total')