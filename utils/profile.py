import numpy as np
from matplotlib import pyplot as plt

def fit_1d(x,y,model="gaussian"):
    import lmfit
    from lmfit.models import GaussianModel,ConstantModel,LorentzianModel
    if model == "lorentzian":
        model = LorentzianModel
    else:
        model = GaussianModel
    mod = model()+ConstantModel()
    pars = model().guess(y, x=x)
    pars =  pars+ConstantModel().guess(y,x=x)
    out  = mod.fit(y, pars, x=x)
    return out


def analize_2dprofile(x,y,img,model="gaussian"):
    fig=plt.figure()
    ax0 = fig.add_axes( [0.1,0.1,0.5,0.5] )
    ax0.pcolormesh(x,y,img)
    def fit(x,y):
        out = fit_1d(x,y,model=model)
        print(out.params)
        return out.best_fit,out.params["fwhm"].value

    ax = fig.add_axes( [0.6,0.1,0.3,0.5],sharey=ax0 )
    ax.plot( img.mean(0),y.mean(0) )
    f1,fwhm1 = fit( y.mean(0),img.mean(0) )
    ax.plot( f1,y.mean(0) )
    ax.set_title("fwhm = %.1f" % fwhm1)

    ax = fig.add_axes( [0.1,0.6,0.5,0.3],sharex=ax0 )
    ax.plot( x.mean(1),img.mean(1) )
    f2,fwhm2 = fit( x.mean(1),img.mean(1) )
    ax.plot( x.mean(1),f2 )
    ax.set_title("fwhm = %.1f" % fwhm2)
    return 

