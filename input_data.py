import scipy.io as sio

data = sio.loadmat('./data/2dPoissonData.mat')
Sig_set_reshape = data["Sig_set_reshape"]
dis_set_reshape = data["dis_set_reshape"]
N_test = data["N_test"]
U_ob_reshape = data["U_ob_reshape"]
def input_data(test):
    if test:
        x = Sig_set_reshape[int(N_test[0][0] * 0.8):]
        x = x.reshape(-1,64,64,1)
        d = dis_set_reshape[int(N_test[0][0] * 0.8):]
        d = d.reshape(-1,64,64,1)
        y = U_ob_reshape[int(N_test[0][0] * 0.8):]
        y = y.reshape(-1,32,32,1)
    else:
        x = Sig_set_reshape[:int(N_test[0][0] * 0.8)]
        x = x.reshape(-1,64,64,1)
        d = dis_set_reshape[:int(N_test[0][0] * 0.8)]
        d = d.reshape(-1,64,64,1)
        y = U_ob_reshape[:int(N_test[0][0] * 0.8)]
        y = y.reshape(-1,32,32,1)
    return x,d,y
