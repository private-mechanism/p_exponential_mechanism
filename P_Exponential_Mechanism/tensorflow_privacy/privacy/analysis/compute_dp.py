import numpy as np
from mpmath import mp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr, Global


wlpath = "C:\\Program Files\\Wolfram Research\\Mathematica\\12.0\\MathKernel.exe"
def compute_dp(lam):
    session = WolframLanguageSession(wlpath)
    session.evaluate(wlexpr('''
       randomgamma[alpha_, beta_, gamma_, samples_] := RandomVariate[GammaDistribution[alpha, beta, gamma, 0], samples];
     '''))
    random_gamma = session.function(wlexpr('randomgamma'))

    session.evaluate(wlexpr('''
       integrant[p_, beta_, d_, Delta_, lam_, r_] := Mean[NIntegrate[
       Sin[x]^(d-2)*Gamma[d/2]/(Sqrt[Pi]*Gamma[(d-1)/2])*(0.99+0.01*Exp[(r^p-(r^2+Delta^2+2*r*Delta*Cos[x])^(p/2))/beta])^(-lam),{x,0,Pi}
       ]];
     '''))
    integrant_moment = session.function(wlexpr('integrant'))
    samples = random_gamma(d / p, beta ** (1 / p), p, num_samples)
    # print(samples)
    moment = integrant_moment(p, beta, d, Delta, lam, samples)
    # print(moment)
    eps = (T * mp.log(moment) + mp.log(1 / delta)) / lam
    session.terminate()
    return eps

# def Bisection1(start,end,privacy):
#     i = 0
#     print(i,(start,end))
#     if privacy(start)<privacy(start+5):
#         lam_list = [start + i for i in range(5)]
#         print(lam_list)
#         privacy_list = [compute_dp(lam) for lam in lam_list]
#         index = privacy_list.index(min(privacy_list))
#         opt_lam = lam_list[index]
#         privacy = np.min(np.array(privacy_list))
#         return opt_lam, privacy
#     else:
#         while ((end-start)>10):
#             i+=1
#             temp = int((start + end) / 2)
#             if (privacy(temp) > privacy(temp +(end-start)/5)):
#                 start = temp
#             else:
#                 end = temp
#             print(i,(start, end))
#         lam_list=[start+i for i in range(end-start+int((end-start)/5))]
#         print (lam_list)
#         privacy_list=[compute_dp(lam) for lam in lam_list]
#         index=privacy_list.index(min(privacy_list))
#         opt_lam=lam_list[index]
#         privacy=np.min(np.array(privacy_list))
#         return opt_lam, privacy

def Bisection(start,end,privacy):
    i = 0
    print(i,(start,end))
    if privacy(start)<privacy(start+5):
        lam_list = [start + i for i in range(5)]
        print(lam_list)
        privacy_list = [compute_dp(lam) for lam in lam_list]
        index = privacy_list.index(min(privacy_list))
        opt_lam = lam_list[index]
        privacy = np.min(np.array(privacy_list))
        return opt_lam, privacy
    else:
        while ((end-start)>5):
            i+=1
            temp = int((start + end) / 2)
            Middle=privacy(temp)
            left=privacy(temp -(end-start)/4)
            right=privacy(temp +(end-start)/4)
            if (Middle< left and Middle < right):
                start = temp -(end-start)/4
                end = temp +(end-start)/4
            if (Middle>left and Middle < right):
                end = temp
            if (Middle<left and Middle > right):
                start=temp
            else:
                print ('there must be an error happended')
            print(i,(start, end))
        lam_list=[start+i for i in range(end-start+int((end-start)/5))]
        print (lam_list)
        privacy_list=[compute_dp(lam) for lam in lam_list]
        index=privacy_list.index(min(privacy_list))
        opt_lam=lam_list[index]
        privacy=np.min(np.array(privacy_list))
        return opt_lam, privacy

if __name__=="__main__":
    p = 0.5
    beta = 2 * 16*16
    d = 5000
    Delta = 1
    T = 10 ** 4
    delta = 10 ** (-5)
    num_samples = 2*10 ** 3
    print(Bisection(1,10**4,compute_dp))

# def privacy(lamda):
#     #这个函数用于计算给定lamda的隐私预算
#
#
#     return lamda**2-582*lamda+7
#
# def get_normalization(dimension,exponent,beta):
#     #这个函数用于计算p-指数分布的归一化项
#     if dimension %2==0:
#         A_d_aid = [mp.pi / (dimension / 2 - i) for i in range(int(dimension / 2))]
#         del A_d_aid[0]
#         A_d = 2 * np.prod(np.array(A_d_aid)) * mp.pi
#
#     else:
#         A_d=2*(mp.pi**(dimension/2))/mp.gamma(dimension/2)
#     alpha=(A_d/exponent)*(beta**(dimension/exponent))*mp.gamma(dimension/exponent)
#     return _to_np_float64(alpha)
#
# def integral_2d(d,p,delta,beta,lamda):
#     #这个函数用于进行二重积分,以计算
#     val2, err2 = mp.quad(lambda r,x:(mp.sin(x)**(d-2)/(0.99+0.01*mp.exp(r**p-(r**2+delta**2-2*r*delta*mp.cos(x))**(p/2)/beta))**lamda)*mp.exp(-r**p/beta)*r**(d-1),  # 函数
#                          [0, mp.inf],  # x上界pi
#                          [0, pi])  # y上界2*
#     return val2, err2


