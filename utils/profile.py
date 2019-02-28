import numpy as np
from matplotlib import pyplot as plt
from datastorage import DataStorage

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


def analyze_2dprofile(x,y,img,model="gaussian",plot=True):

    if x.ndim == 2: x=x[:,0]
    if y.ndim == 2: y=y[0]

    def fit(x,intensity):
        out = fit_1d(x,intensity,model=model)
        print(out.params)
        return out,out.best_fit,out.params["fwhm"].value

    ox,fx,fwhmx = fit( x,img.mean(1) )
    oy,fy,fwhmy = fit( y,img.mean(0) )

    if plot:
        fig=plt.figure()
        ax0 = fig.add_axes( [0.1,0.1,0.5,0.5] )
        ax0.pcolormesh(x,y,img)

        ax = fig.add_axes( [0.6,0.1,0.3,0.5],sharey=ax0 )
        ax.plot( img.mean(0),y )
        ax.plot( fy,y )
        ax.set_title("fwhm = %.1f" % fwhmy)

        ax = fig.add_axes( [0.1,0.6,0.5,0.3],sharex=ax0 )
        ax.plot( x ,img.mean(1) )
        ax.plot( x,fx )
        ax.set_title("fwhm = %.1f" % fwhmx)
    out = DataStorage( x=x,y=y,data=img,fitx=ox,fity=oy)
    return out

