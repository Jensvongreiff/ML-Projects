import numpy as np
import preprocessing as pp
import generator as gen

review_1, review_5 = pp.load_dataset("task_01_data.npy")
test = gen.HMM_Gen(review_5, 8)
sequence = [1,6,29,2,80]
# forward_test = test.forwards(sequence)
# backward_test = test.backwards(sequence)
# gamma, chi, indices = test.E_step(['to','I', 'and'])
# test.M_step(gamma, chi, indices)
test.train(50, plot=True)
print('shape A: ' + str(np.shape(test.parameters.A)) + ' shape B: ' + str(np.shape(test.parameters.B)) + ' shape pi: ' + str(np.shape(test.parameters.pi)))
print("\n")
for i in range(5):
    print(test.generate_sentence(7))
