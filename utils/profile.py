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

    projx = img.mean(0)
    projy = img.mean(1)

    ox,fx,fwhmx = fit( x,projx )
    oy,fy,fwhmy = fit( y,projy )

    if plot:
        fig=plt.figure()
        ax0 = fig.add_axes( [0.1,0.1,0.5,0.5] )
        ax0.pcolormesh(x,y,img)

        ax = fig.add_axes( [0.6,0.1,0.3,0.5],sharey=ax0 )
        ax.plot( projy,y )
        ax.plot( fy,y )
        ax.set_title("fwhm = %.1f" % fwhmy)

        ax = fig.add_axes( [0.1,0.6,0.5,0.3],sharex=ax0 )
        ax.plot( x ,projx )
        ax.plot( x,fx )
        ax.set_title("fwhm = %.1f" % fwhmx)
    out = DataStorage( x=x,y=y,data=img,fitx=ox,fity=oy)
    return out

def transmission_trough_pinhole(opening,intensity=1,beam_fwhm=1):
    """ Transmission of a gaussian beam trough a pinhole of a given size """
    sigma = beam_fwhm/2.35482004503 # sqrt(8*ln2)
    return intensity*(1-np.exp(-opening**2/2/sigma**2))

def fit_transmission_trough_pinhole(opening,intensity,fix_intensity=False,plot=True):
    import lmfit
    model = lmfit.Model(transmission_trough_pinhole)
    pars = model.make_params( intensity=intensity.max(), beam_fwhm=np.mean(opening))
    if fix_intensity: pars['intensity'].vary = False
    res = model.fit(intensity,x=opening)
    if plot:
        res.plot()
    return res




