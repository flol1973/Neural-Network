input = [1 ,2 ,3,2.5]

weight1 = [-0.2,0.4,0.7,0.6]
weight2 = [0.2,-0.4,0.7,0.6]
weight3 = [0.22,0.4,-0.7,0.6]

bias1 = 3
bias2 = -3
bias3 = 3

output = [input[0]*weight1[0] + input[1]*weight1[1] + input[2]*weight1[2] +bias1,
          input[0]*weight2[0] + input[1]*weight2[1] + input[2]*weight2[2] +bias2,
          input[0]*weight3[0] + input[1]*weight3[1] + input[2]*weight3[2] +bias3] 

print(output)