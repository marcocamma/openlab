from __future__ import division, print_function

import codecs
import collections
import functools
import os
import re
import string
import sys
import types
from itertools import chain

# import sys
# if (sys.version_info[0] < 3):
import numpy as np
import pylab as plt
import scipy.signal
from numpy import exp

SQRT2 = np.sqrt(2)

_svd_ret = collections.namedtuple("svd_return", "basis s population data")

# COLORS, ETC ###
colors = None

nice_colors = (
    # from http://www.mulinblog.com/a-color-palette-optimized-for-data-visualization/
    "#4D4D4D",  # (gray)
    "#5DA5DA",  # (blue)
    "#FAA43A",  # (orange)
    "#60BD68",  # (green)
    "#F17CB0",  # (pink)
    "#B2912F",  # (brown)
    "#B276B2",  # (purple)
    "#DECF3F",  # (yellow)
    "#F15854",  # (red)
)


nice_colors = ['#014636',
               '#016c59',
               '#02818a',
               '#3690c0',
               '#67a9cf',
               '#a6bddb',
               '#d0d1e6',
               '#ece2f0']


def colormap(list_of_colors):
    from matplotlib.colors import colorConverter, LinearSegmentedColormap
    c = [colorConverter.to_rgba(l) for l in list_of_colors]
    # make the colormaps
    cmap = LinearSegmentedColormap.from_list('cmap', c, 256)
    return cmap


def simpleaxis(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def noaxis(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def pulseDuration(t0, L, GVD):
    return t0*np.sqrt(1.+(L/(t0**2/2/np.abs(GVD))))


class MCError(Exception):
    """ Exception handler for this library """
    def __init__(self, value):
        super().__init__(self,value)
        self.value = value

    def __str__(self):
        return repr(self.value)

# OPTICS ###


def beam_displacement(n=1.4, thickness=1, angle=20):
    """ calcualte beam dispacement for a slab of given thickness """

    # angle inside
    angle = np.deg2rad(angle)
    angle_in = np.arcsin(np.sin(angle)/n)
    dy_angle = thickness*np.sin(angle)
    dy_angle_in = thickness*np.sin(angle_in)
    return dy_angle_in-dy_angle


class GaussianBeam(object):
    """ w is the beam radius (defined as half of the full width @ 1/e2)
        Etot is the area integrated intensity
        lam and w|fwhm have to have the same dimensions (mm,m,nm)
    """

    def __init__(self, w=None, fwhm=None, Etot=1, lam=1):
        assert w is not None or fwhm is not None, "fwhm or w have to be given"
        assert w is None or fwhm is None, "fwhm or w have to be given, not both"
        if fwhm is not None:
            w = fwhm/np.sqrt(2)/np.sqrt(np.log(2))
        self.w = w
        self.Etot = Etot
        self.lam = lam

    @property
    def fwhm(self):
        return self.w*np.sqrt(2)*np.sqrt(np.log(2))

    def peak_fluence(self):
        """ for r=0, see Beam_diameter on wikipedia and gaussian_beams on rp_photonics """
        return 2*self.Etot/np.pi/self.w**2

    @property
    def F0(self): return self.peak_fluence()

    @property
    def fwhm_fluence(self): return self.Etot/self.fwhm**2

    @property
    def rayleigh(self):
        return np.pi*self.w**2/self.lam

    @property
    def far_field_half_angle(self):
        return self.lam/np.pi/self.w

# VECTOR ... ETC. ###


def vectorLenght(v):
    """ assuming axis -1 as coord """
    return np.sqrt(np.sum(v*v, axis=-1))


def versor(v):
    return v/vectorLenght(v)[:, np.newaxis]


# LIST,INDEXING ... ETC. ###

def smartIdx(idx, forceContigous=False):
    """ Try to interpret an array of bool as slice;
    this allows selecting a subarray alot more efficient
    since array[slice] it returns a view and not a copy """
    if isinstance(idx, int):
        ret = slice(idx, idx+1)
    elif isinstance(idx, slice):
        ret = idx
    else:
        idx = np.asarray(idx)
        if idx.dtype == np.bool:
            i = np.where(idx)[0]
        else:
            i = idx
        # in case there is only one
        if (len(i) == 1):
            ret = slice(i[0], i[0]+1)
            return ret
        if forceContigous:
            ret = slice(i[0], i[-1])
        else:
            d = i[1:]-i[0:-1]
            dmean = int(d.mean())
            if np.all(d == dmean):
                ret = slice(i[0], i[-1]+1, dmean)
            else:
                ret = idx
    return ret


# CONVOLUTION,INTERPOLATION,SMOOTHING ... ETC. ###

def poly_approximant(x, y, order=10, allowExtrapolation=False, fill_value=0):
    """ return a polinomial view """
    poly = np.polyfit(x, y, order)

    def f(xx):
        res = np.polyval(poly, xx)
        if allowExtrapolation:
            return res
        else:
            if np.isscalar(xx) and ((xx > x.max()) or (xx < x.min())):
                return fill_value
            elif not np.isscalar(xx):
                idx = (xx < x.min()) | (xx > x.max())
                res[idx] = fill_value
            return res
    return f


def svd(data, ncomp=None, truncate=True):
    """ do SVD based cleaning

        Parameters
        ----------
        ncomp : int
          number of components to keep
        truncate : bool
          if True, returns only basis and populations of up to ncomp
    """
    # first index is time/T/P
    u, s, v = np.linalg.svd(data, full_matrices=False)
    if ncomp is not None:
        s[ncomp:] = 0
        data = np.dot(u, np.dot(np.diag(s), v))
        if truncate:
            v = v[:ncomp]
            u = u[:, :ncomp]
    return _svd_ret(basis=v, s=s, population=u.T, data=data)


def smoothing(x, y, err=None, k=5, s=None, newx=None, derivative_order=0):
    idx = np.isnan(x) | np.isnan(y)
    idx = ~ idx
    if newx is None:
        newx = x
    if idx.sum() > 0:
        x = x[idx]
        y = y[idx]
    if idx.sum() < 3:
        return np.ones(len(newx))
    if err is None:
        w = None
    elif err == "auto":
        n = len(x)
        imin = max(0, n/2-20)
        imax = min(n, n/2+20)
        idx = range(imin, imax)
        p = np.polyfit(x[idx], y[idx], 4)
        e = np.std(y[idx] - np.polyval(p, x[idx]))
        w = np.ones_like(x)/e
    else:
        w = np.ones_like(x)/err
    from scipy.interpolate import UnivariateSpline
    if s is not None:
        s = len(x)*s
    s = UnivariateSpline(x, y, w=w, k=k, s=s)
    if (derivative_order == 0):
        return s(newx)
    else:
        try:
            len(derivative_order)
            return [s.derivative(d)(newx) for d in derivative_order]
        except:
            return s.derivative(derivative_order)(newx)


def interpolator(x, y, kind='linear', axis=-1, copy=False, bounds_error=False, fill_value=np.nan):
    from scipy import interpolate
    if (kind != "linear"):
        print("Warning interp1d can be VERY SLOW when using something that is not liear")
    f = interpolate.interp1d(x, y, kind=kind, axis=axis, copy=copy,
                             bounds_error=bounds_error, fill_value=fill_value)
    return f


def interpolator_spl(x, y, kind="cubic"):
    from scipy import interpolate as itp
    if kind == "linear":
        kind = 1
    if kind == "cubic":
        kind = 3
    splinepars = itp.splrep(x, y, k=kind)

    def f(x, der=0,):
        """f(x) returns values for x[i]. f(x,order) return order-th derivative"""
        return itp.splev(x, splinepars, der=der)
    return f


def interpolate_fast(x, y, newx, kind='cubic'):
    f = interpolator_spl(x, y, kind=kind)
    return f(newx)


def interpolate(x, y, newx, kind='linear', axis=-1, copy=False,
                bounds_error=False, fill_value=np.nan):
    f = interpolator(x, y, kind=kind, axis=axis, copy=copy,
                     bounds_error=bounds_error, fill_value=fill_value)
    return f(newx)


def getElement(a, i, axis=-1):
    nDim = a.ndim
    if (axis < 0):
        axis = nDim+axis
    colon = (slice(None),)
    return a[colon*axis+(i,)+colon*(nDim-axis-1)]


def setElement(a, i, res, axis=-1):
    temp = getElement(a, i, axis=axis)
    np.copyto(temp, res)


def invertedView(x):
    return x[slice(None, None, -1)]


def convolveQuad(x, y, xres, yres, useAutoCrop=True,
                 fill_value=0., axis=-1):
    print("NOT FINISHED")
    return
    from scipy import integrate
    if useAutoCrop:
        idxRes = smartIdx(yres > (1e-4*yres.max()))
        # print("Autocropping",idxRes,xres.shape)
    else:
        idxRes = slice(None)
    xresInv = -invertedView(xres[idxRes])
    yresInv = invertedView(yres[idxRes])
    area = integrate.simps(yres, x=xres)

    f = interpolator(x, y)
    r = interpolator(xres, yres)

    if y.ndim < 2:
        y = y[np.newaxis, :]
    nDim = y.ndim
    if (axis < 0):
        axis = nDim+axis

    # expand yres to allow broadcasting
    sh = [1, ]*y.ndim
    sh[axis] = len(yres)
    yres = yres.reshape(sh)

    # prepare output
    out = np.empty_like(y)
    nP = len(x)
    for i in range(nP):
        # interpolate x,y on xres+x[i]
        x_integral = xresInv+x[i]
        ytemp = interpolate(
            x, y, x_integral, fill_value=fill_value, axis=axis)/area
        # do integration
        res = integrate.simps(ytemp*yresInv, x=xresInv+x[i], axis=axis)
        colon = (slice(None),)
        setElement(out, i, res, axis=axis)
    return out


def convolve(x, y, xres, yres, useAutoCrop=True, approximantOrder=None,
             fill_value=0., axis=-1):
    """ if approximantOrder is not None, use interpolating polynomial of order
        approximantOrder to perform integration """
    from scipy import integrate
    import copy
    if useAutoCrop:
        idxRes = smartIdx(yres > (1e-4*yres.max()))
        # print("Autocropping",idxRes,xres.shape)
    else:
        idxRes = slice(None)
    xresInv = -invertedView(xres[idxRes])
    yresInv = invertedView(yres[idxRes])
    area = integrate.simps(yres, x=xres)

    if approximantOrder is not None:
        # print("Using approximant!!",xresInv.shape)
        approx = poly_approximant(xresInv, yresInv/area, approximantOrder,
                                  allowExtrapolation=False, fill_value=0)
        # return approx,xresInv,yresInv
        return convolveFunc(x, y, approx, fill_value=fill_value, axis=axis,)

    if y.ndim < 2:
        y = y[np.newaxis, :]
    nDim = y.ndim
    if axis < 0:
        axis = nDim+axis

    # expand yres to allow broadcasting
    sh = [1, ]*y.ndim
    sh[axis] = len(yres)
    yres = yres.reshape(sh)

    # prepare output
    out = np.empty_like(y)
    nP = len(x)
    f = interpolator(x, y, fill_value=fill_value, axis=axis)
    for i in range(nP):
        # interpolate x,y on xres+x[i]
        x_integral = xresInv+x[i]
        ytemp = f(x_integral)/area
        # ytemp  = interpolate(x,y,x_integral,fill_value=fill_value,axis=axis)/area
        # ytemp = f(x_integral)/area
        # do integration
        res = integrate.simps(np.transpose(
            ytemp.T*yresInv), x=xresInv+x[i], axis=axis)
        colon = (slice(None),)
        setElement(out, i, res, axis=axis)
    return out


def fftconvolve(x, y, yres, xres=None, normalize=False):
    if (xres is not None):
        yres = interpolate(xres, yres, x, fill_value=0)
    fft = scipy.signal.fftconvolve(y, yres, "full")
    _idx = np.argmin(np.abs(x))  # find t=0
    fft = fft[_idx:_idx+len(x)]
    if normalize:
        norm = fftconvolve_find_norm(x, yres, xres=None)
    else:
        norm = 1
    return fft/norm


def fftconvolve_find_norm(x, res, xres=None):
    step = np.ones_like(x)
    n = int(len(step)/2)
    step[:n] = 0
    norm = fftconvolve(x, step, res, xres=xres, normalize=False).max()
    return norm


def convolveGaussian(x, y, sig=1., nPointGaussian=51, fill_value=0., axis=-1):
    xG = np.linspace(-5*sig, 5*sig, nPointGaussian)
    G = gaussian(xG, x0=0, sig=sig)
    return convolve(x, y, xG, G, fill_value=fill_value, axis=axis)


def convolveFunc(x, y, func_res, fill_value=0., axis=-1):
    from scipy import integrate
    if y.ndim < 2:
        y = y[np.newaxis, :]
    nDim = y.ndim
    if (axis < 0):
        axis = nDim+axis

    # prepare output
    out = np.empty_like(y)

    nP = len(x)
    for i in range(nP):
        # calculate the values of the resolution function on x-x[i]
        ytemp = func_res(x-x[i])
        # do integration
        res = integrate.simps(y*ytemp, x=x-x[i], axis=axis)
        setElement(out, i, res, axis=axis)
    return out


def convolveFuncParams(x, y, func_res, func_res_pars, fill_value=0., axis=-1):
    def tempFunc(xx):
        return func_res(xx, *func_res_pars)
    return convolveFunc(x, y, tempFunc, fill_value=fill_value, axis=axis)


def convolve_test(nG=51):
    import time
    x = np.arange(100)
    y = (x > 50).astype(np.float)
    sig = 3
    xG = np.linspace(-5*sig, 5*sig, nG)
    G = gaussian(xG, x0=0, sig=sig)
    Gpoly = np.polyfit(xG, G, 20)
    t0 = time.time()
    conv_num = convolve(x, y, xG, G, fill_value=0.)
    print("Num:", time.time()-t0)
    t0 = time.time()
    conv_poly = convolve(x, y, xG, G, approximantOrder=10, fill_value=0.)
    print("Num:", time.time()-t0)
    t0 = time.time()
    conv_fun = convolveFuncParams(x, y, gaussian, (0, sig), fill_value=0.)
    print("Fun:", time.time()-t0)
    import pylab as plt
    plt.plot(conv_num.T, label="Numerical %d points" % nG)
    plt.plot(conv_fun.T, "o", label="Gaussian F")
    plt.plot(conv_poly.T, label="Gaussian (poly)")
    plt.plot(conv_num.T-conv_fun.T)
    return conv_num, conv_fun


def conv_gauss_and_const(x, sig):
    from scipy.special import erf
    return 0.5*(1-erf(-x/SQRT2/sig))


def conv_gauss_and_exp(x, sig, tau):
    from scipy.special import erf
    # from mpmath import erf
    # http://www.numberempire.com/integralcalculator.php?function=exp%28-x%2Fl%29*exp%28-%28t-x%29**2%2F2%2Fs**2%29%2FSqrt%282%29%2FSqrt%28pi%29%2Fs&var=x&answers=
    # actually calculated with sympy
    # return -(erf(SQRT2*(sig**2 - x*tau)/(2*sig*tau)) - 1)*exp(sig**2/(2*tau**2) - x/tau)/2
    return 0.5*np.exp(-(2*tau*x-sig**2)/2/tau**2)*(1-erf((-tau*x+sig**2)/SQRT2/tau/sig))


def gaussian(x, x0=0, sig=1, normalize=True):
    g = np.exp(-(x-x0)**2/2/sig**2)
    if normalize:
        return 1/np.sqrt(2*np.pi)/sig*g
    else:
        return g


# FFT filter ##
class FFTfilter(object):
    def __init__(self, s, dx=1, wins=((0.024, 0.01),), wintype="gauss"):
        f, ffts = fft(s, dx=dx)
        filter = np.ones_like(f)
        for win in wins:
            if wintype == "gauss":
                filter *= (1-gaussian(f, win[0], win[1], normalize=False))
                filter *= (1-gaussian(f, -win[0], win[1], normalize=False))
            else:
                print("Not implemented")

        self.filter = filter

    def apply(self, s):
        s = np.fft.fft(s)
        return np.real(np.fft.ifft(s*self.filter))


# FFT filter ##
class FFTlowpass(object):
    def __init__(self, s, dx=1, fmax=1e3):
        if s.ndim >1: s = s[0]
        f, ffts = fft(s, dx=dx)
        filter = np.ones_like(f)
        idx = np.abs(f) > fmax
        filter[idx]=0

        self.filter = filter

    def apply1d(self, s):
        s = np.fft.fft(s)
        return np.real(np.fft.ifft(s*self.filter))

    def apply(self, s):
        if s.ndim == 2:
            out = np.empty_like(s)
            for i,d in enumerate(s):
                out[i] = self.apply1d(d)
        else:
            out = self.apply1d(s)
        return out


# OS, PROMPT,PROCESS,TIME,DATE ... ETC. ###

def bytesToHuman(bytes, units="auto", fmt="%.2f %s"):
    _units = dict(B=0, KB=1, MB=2, GB=3, TB=4, PT=5)
    _symb = {v: k for k, v in _units.items()}
    bytes = float(bytes)
    if units == "auto":
        u = np.log(bytes)//np.log(1024)
        units = _symb[u]
    else:
        u = _units[units]
    value = bytes/1024**u
    return fmt % (value, units)


_time_regex = re.compile(r"(-?\d+\.?\d*)((?:s|fs|ms|ns|ps|us)?)")


def strToTime(delay):
    _time2value = dict(fs=1e-15, ps=1e-12, ns=1e-9, us=1e-6, ms=1e-3, s=1)

    match = _time_regex.search(delay)
    if match:
        n, t = float(match.group(1)), match.group(2)
        value = _time2value.get(t, 1)
        return n*value
    else:
        return None


def memAvailable(asHuman=True):
    import psutil
    m = psutil.virtual_memory()  # return bytes
    m = m.available
    if asHuman:
        m = bytesToHuman(m)
    return m


class procCom(object):
    def __init__(self, cmd):
        import pexpect
        self.proc = pexpect.spawn(cmd)

    def get(self, timeout=None, waitFor=None):
        import pexpect
        if (waitFor is not None):
            s = ""
            try:
                while s.find(waitFor) < 0:
                    s += self.proc.read_nonblocking(timeout=timeout)
            except (pexpect.TIMEOUT, pexpect.EOF):
                pass
        else:
            s = ""
            try:
                while 1:
                    s += self.proc.read_nonblocking(timeout=timeout)
            except (pexpect.TIMEOUT, pexpect.EOF):
                pass
        # print "got",s
        return s

    def send(self, what):
        self.proc.write(what)
        self.proc.flush()
        # print "send",what

    def query(self, what, timeout=None, waitFor=None):
        self.send(what)
        return self, get(timeout=timeout, waitFor=waitFor)


def getCMD(cmd, strip=True):
    import os
    shell = os.popen(cmd)
    ret = shell.readlines()
    shell.close()
    if (strip):
        ret = [x.strip() for x in ret]
    return ret


def lsdir_old(path, withQuotes=False, recursive=False):
    if recursive:
        dirs = []
        for (dir, _, file) in os.walk(path):
            dirs.append(dir)
    else:
        content = getCMD("ls -1 %s" % path)
        content = ["%s/%s" % (path, x) for x in content]
        dirs = [x for x in content if os.path.isdir(x)]
    if (withQuotes):
        dirs = ["'%s'" % x for x in dirs]
    return dirs


def lsdir(folder="./", withQuotes=False, recursive=False):
    if not recursive:
        folders = list(os.scandir(folder))
        folders = [f.path for f in folders if f.is_dir()]
    else:
        folders = []

        def finddirs(path):
            inside = list(os.scandir(path))
            inside = [f.path for f in inside if f.is_dir()]
            folders.append(path)
            [finddirs(f) for f in inside]
        finddirs(folder)
    if (withQuotes):
        folders = ["'%s'" % x for x in folders]
    return folders


def lsfiles(path, withQuotes=False, recursive=False):
    if recursive:
        print("Not sure is working")
        files = []
        for (dir, _, file) in os.walk(path):
            files.append(file)
    else:
        content = getCMD("ls -1 %s" % path)
        content = ["%s/%s" % (path, x) for x in content]
        files = [x for x in content if os.path.isfile(x)]
    if (withQuotes):
        files = ["'%s'" % x for x in files]
    return files


def downloadPDB(pdbID, outFileName=None):
    import urllib2
    import os
    address = "http://www.rcsb.org/pdb/download/downloadFile.do?fileFormat=pdb&compression=NO&structureId=%s" % pdbID
    p = urllib2.urlopen(address)
    lines = p.readlines()
    if (outFileName is None):
        outFileName = pdbID+".pdb"
    folder = os.path.dirname(outFileName)
    if (folder != '' and not os.path.isdir(folder)):
        os.makedirs(folder)
    f = open(outFileName, "w")
    f.write("".join(lines))
    f.close()


def dateStringToObj(s, format="%Y.%m.%d %H:%M:%S"):
    import datetime
    return datetime.datetime.strptime(s, format)


def now():
    import time
    return time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())


def mytimer(func, args):
    import time
    t0 = time.time()
    ret = func(*args)
    return time.time()-t0, ret

# TXT I/O ###


def lineToVals(line):
    return map(eval, string.split(line))


class DataFile(object):
    filename = ""
    lines = []
    comments = []
    head = {}
    Data = []
    Ndata = 0

    def __init__(self, f):
        self.filename = f
        self.Read()

    def Read(self):
        f = open(self.filename, "r")
        temp = f.readlines()
        f.close()
        r = re.compile(r'\s+')
        c = re.compile(r'^\s*#')
        for l in temp:
            if (re.match(c, l)):
                self.comments.append(l)
            else:
                self.lines.append(l)
        # first of non commented line might be header
        # try to understand it
        keys = []
        try:
            v = lineToVals(self.lines[0])
            for i in range(len(v)):
                keys.append(i)
        except:
            keys = self.lines[0].split()
            self.lines.remove(self.lines[0])
        datatemp = []
        for l in self.lines:
            datatemp.append(lineToVals(l))
        self.Data = np.asarray(datatemp)
        for i in range(len(keys)):
            self.head[keys[i]] = self.Data[:, i]
            self.__dict__[keys[i]] = self.Data[:, i]
        (self.Ndata, self.Ncol) = self.Data.shape

    def clean(self):
        del self.Data


def writev(fname, x, Ys, form="%+.6g", sep=" ", header=None, headerv=None):
    """ Write data to file 'fname' in text format.
        Inputs:
          x = x vector
          Ys = vector(or array or list of vectors) for the Ys
          form = format to use
          sep = separator
          header = text header (must be a string)
          headerv = vector to be used as header, it is convienient when
            the output must be of the form
              Ncol 252 253 254
              x1   y11 y12 y13
              .......
            In this case headerv should be [252,253,254]
    """
    if (type(x) != np.ndarray):
        x = np.array(x)
    if (type(Ys) != np.ndarray):
        Ys = np.array(Ys)
    if (len(Ys.shape) == 1):
        Ys = Ys.reshape(Ys.shape[0], 1)
    nx = len(x)
    if (Ys.shape[0] == nx):
        ny = Ys.shape[1]
    elif (Ys.shape[1] == nx):
        ny = Ys.shape[0]
        Ys = np.transpose(Ys)
    else:
        raise MCError("dimension of x (%d) does not match any of the dimensions of Ys (%d,%d)" % (
            nx, Ys.shape[0], Ys.shape[1]))
    f = codecs.open(fname, encoding='utf-8', mode="w")
    if (header is not None):
        f.write(header.strip()+"\n")
    if (headerv is not None):
        f.write("%d" % (ny+1))
        for i in range(ny):
            f.write(sep)
            f.write(form % headerv[i])
        f.write("\n")
    for i in range(nx):
        f.write(form % x[i])
        f.write(sep)
        for j in range(ny-1):
            f.write(form % Ys[i, j])
            f.write(sep)
        f.write(form % Ys[i, -1])
        f.write("\n")
    f.close()


def writev_numpy(fname, x, Ys, form="%+.6g", sep=" ", header=None, headerv=None):
    """ Write data to file 'fname' in text format.
        Inputs:
          x = x vector
          Ys = vector(or array or list of vectors) for the Ys
          form = format to use
          sep = separator
          header = text header (must be a string)
          headerv = vector to be used as header, it is convienient when
            the output must be of the form
              Ncol 252 253 254
              x1   y11 y12 y13
              .......
            In this case headerv should be [252,253,254]
    """
    if (type(x) != np.ndarray):
        x = np.array(x)
    if (type(Ys) != np.ndarray):
        Ys = np.array(Ys)
    if (len(Ys.shape) == 1):
        Ys = Ys.reshape(Ys.shape[0], 1)
    nx = len(x)
    if (Ys.shape[1] == nx):
        ny = Ys.shape[0]
    elif (Ys.shape[0] == nx):
        ny = Ys.shape[1]
        Ys = np.transpose(Ys)
    else:
        raise MCError("dimension of x (%d) does not match any of the dimensions of Ys (%d,%d)" % (
            nx, Ys.shape[0], Ys.shape[1]))
    f = codecs.open(fname, encoding='utf-8', mode="w")
    if (header is not None):
        f.write(header.strip()+"\n")
    if (headerv is not None):
        f.write("%d" % (ny+1))
        for i in range(ny):
            f.write(sep)
            f.write(form % headerv[i])
        f.write("\n")
    out = np.vstack((x, Ys))
    np.savetxt(f, np.transpose(out), fmt=form, delimiter=sep)


def loadtxt(fname, hasVectorHeader=True, asObj=False, isRecArray=False,skiprows=0,delimiter=None):
    if (isRecArray):
        return loadRecArray(fname, asObj=asObj)
    a = np.loadtxt(fname,skiprows=skiprows,delimiter=delimiter)
    if (not hasVectorHeader):
        x = a[:, 0]
        y = a[:, 1:].T
        t = None
    else:
        x = a[1:, 0]
        y = a[1:, 1:].T
        t = a[0, 1:]
    if (asObj):
        class txtFile(object):
            def __init__(self, x, y, t):
                self.x = x
                self.y = y
                self.t = t
        return txtFile(x, y, t)
    else:
        return x, y, t


def loadRecArray(fname, hasVectorHeader=True, asObj=False):
    a = np.loadtxt(fname, skiprows=1)
    # find header
    f = open(fname, "r")
    found = None
    while found is None:
        s = f.readline().strip()
        if s[0] != "#":
            found = s
    names = found.split()
    if (asObj):
        class txtFile(object):
            def __init__(self):
                pass
        ret = txtFile()
        for i in range(len(names)):
            ret.__dict__[names[i]] = a[:, i]
    else:
        ret = np.core.records.fromarrays(a.transpose(), names=",".join(names))
    return ret


def dictToRecArray(mydict):
    shapes = np.asarray([v.shape[0] for v in mydict.values()])
    assert np.all(shapes == shapes[0])
    names = list(mydict.keys())
    formats = [mydict[p].dtype for p in names]
    arr = np.empty(shapes[0],  dtype={'names': names, 'formats': formats})
    for n in names:
        arr[n] = mydict[n]
    return arr


def prepareRecArrayFromDict(mydict, n=1, leaveEmpty=True):
    names = list(mydict.keys())
    formats = [type(mydict[p]) for p in names]
    if leaveEmpty:
        array_kind = np.empty
    else:
        array_kind = np.zeros
    return array_kind(n, dtype={'names': names, 'formats': formats})


def prepareRecArrayFromNamesAndArray(names, ar, n=1, leaveEmpty=True):
    formats = [ar.dtype.type for p in names]
    if leaveEmpty:
        array_kind = np.empty
    else:
        array_kind = np.zeros
    return array_kind(n, dtype={'names': names, 'formats': formats})


def writeMatrix(fout, M, x, y, form="%+.6g", sep=" ", header=None):
    (nx, ny) = M.shape
    if ((nx == len(x)) and (ny == len(y))):
        pass
    elif ((nx == len(y)) and (ny == len(x))):
        M = M.transpose()
        (nx, ny) = M.shape
    else:
        e = "Dimensions of matrix and the x or y vectors don't match"
        e += "shapes of M, x, y: " + \
            str(M.shape) + " " + str(len(x)) + " " + str(len(y))
        raise MCError(e)
    temp = np.zeros((nx+1, ny))
    temp[1:, :] = M
    temp[0, :] = y
    writev(fout, np.hstack((0, x)), temp, form=form, sep=sep, header=header)


# MATPLOTLIB ... ETC. ###

def lt(i, style="-", colors='rgbk'):
    i = i % len(colors)
    color = colors[i]
    if (style is not None):
        color += style
    return color


def color(i, colors=nice_colors):
    i = i % len(colors)
    color = colors[i]
    return color


def displayFig(i, x=None, y=None, roi=None):
    import pylab as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    fig = plt.figure()
    n1, n2 = i.shape
    if x is None:
        x = np.arange(n1)
    if y is None:
        y = np.arange(n2)

    ax = fig.add_subplot(111)
    if roi is not None:
        (x1, x2, y1, y2) = roi
    else:
        x1, x2, y1, y2 = (0, n1, 0, n2)
    xm = x[x1]
    xM = x[x2-1]
    ym = y[y1]
    yM = y[y2-1]

    def _format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col >= 0 and col < n2 and row >= 0 and row < n1:
            z = i[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    iplot = i[x1:x2, y1:y2]
    im1 = ax.imshow(iplot, origin='bottom', extent=[xm, xM, ym, yM])
    ax.format_coord = _format_coord
    fig.subplots_adjust(left=0.25, bottom=0.25)
    fig.colorbar(im1)
    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
    smin = Slider(axmin, 'Min', -5000, 5000, valinit=-200)
    smax = Slider(axmax, 'Max', -5000, 5000, valinit=200)

    def update(val):
        im1.set_clim([smin.val, smax.val])
        fig.canvas.draw()
    smin.on_changed(update)
    smax.on_changed(update)


def savefig(fname, figsize, fig=None, **kwargs):
    """
      force saving figure with a given size, useful when using tiling wm;
      if fname is a list, it saves multiple files, for example [todel.pdf,todel.png]
    """
    if isinstance(fname, str):
        fname = (fname,)
    if fig is None:
        fig = plt.gcf()
    old_bkg = plt.get_backend()
    old_inter = plt.isinteractive()
    try:
        plt.switch_backend("cairo")
        old_height = fig.get_figheight()
        old_width = fig.get_figwidth()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        [fig.savefig(f, **kwargs) for f in fname]
        plt.switch_backend(old_bkg)
    finally:
        plt.switch_backend(old_bkg)
        plt.interactive(old_inter)

# FFT ###


def fft(y, dx=1):
    fft = np.fft.fft(y)
    f = np.fft.fftfreq(len(y), d=dx)
    return f, fft


def wrap(vec, at_idx):
    """ wrap vector at such that at_idx corresponds to the center
        it might be useful for FFT (should try to implemente it using np.roll)"""
    nhalf = int(len(vec)/2)
    if (at_idx > nhalf):
        vec = np.concatenate((vec[at_idx-nhalf:], vec[:at_idx-nhalf]))
    else:
        vec = np.concatenate((vec[at_idx+nhalf:], vec[:at_idx+nhalf]))
    return vec

# REBIN, RUNNING AVG, ... ETC ###


def rebinOLD(bins_edges, x, *Y):
    """ rebins a list of Y based on x using bin_edges
        returns
          center of bins (vector)
          rebinned Y     (list of vectors)
          Simga's        (list of vectors), note: just std not std of average
          N              (list of vectors), number of occurrances
    """
    n = len(bins_edges)-1
    outX = []
    outY = []
    outS = []
    outN = []
    for j in range(len(Y)):
        outY.append(np.empty(n))
        outS.append(np.empty(n))
        outN.append(np.empty(n))
    outX = np.empty(n)
    for i in range(n):
        idx = (x >= bins_edges[i]) & (x < bins_edges[i+1])
        outX[i] = (bins_edges[i]+bins_edges[i+1])/2.
        print("IDX", i, idx.sum(), outX[i])
        if (idx.sum() > 0):
            for j in range(len(Y)):
                outN[j][i] = idx.sum()
                outY[j][i] = Y[j][idx].mean()
                outS[j][i] = Y[j][idx].std()
        else:
            for j in range(len(Y)):
                outN[j][i] = 0
                outY[j][i] = np.nan
                outS[j][i] = np.nan
    return outX, outY, outS, outN


def rebin1D(a, shape):
    sh = shape, a.shape[0]//shape
    return a.reshape(sh).mean(1)


def rebin1Dnew(a, shape):
    n0 = a.shape[0]//shape
    sh = shape, n0
    return a[:n0*shape].reshape(sh).mean(1)


def rebin2D(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def rebin2Dnew(a, shape):
    # // means floor
    n0 = a.shape[0]//shape[0]
    n1 = a.shape[1]//shape[1]
    crop = shape[0]*n0, shape[1]*n1
    sh = shape[0], n0, shape[1], n1
    # print a[:n0*shape[0],:n1*shape[1]].reshape(sh)
    return a[:crop[0], :crop[1]].reshape(sh).mean(-1).mean(1)


def rebin3Dnew(a, shape):
    # // means floor
    n0 = a.shape[0]//shape[0]
    n1 = a.shape[1]//shape[1]
    n2 = a.shape[2]//shape[2]
    sh = shape[0], n0, shape[1], n1, shape[2], n2
    crop = n0*shape[0], n1*shape[1], n2*shape[2]
    # print a[:n0*shape[0],:n1*shape[1]].reshape(sh)
    return a[:crop[0], :crop[1], :crop[2]].reshape(sh).mean(-1).mean(-2).mean(-3)


def rebin(a, shape):
    a = np.asarray(a)
    ndim = a.ndim
    if (ndim == 1):
        return rebin1Dnew(a, shape)
    elif (ndim == 2):
        return rebin2Dnew(a, shape)
    elif (ndim == 3):
        return rebin3Dnew(a, shape)
    else:
        print("Can't do rebin of", a)


def rebinTODO(a, shape):
    ndim = a.ndim
    if (len(shape) != ndim):
        print("Error, asked to rebin a %d dimensional array but provided shape with %d lengths" % (
            ndim, len(shape)))
        return None
    nout = []
    for n in range(ndim):
        nout.append(a.shape[n] % shape[n])
    print("Not implemented ...")

# DISTRIBUTION, MEDIAN, SIGMA, ETC. ###


def idx_within_std_from_center(vector, range):
    (m, s) = MedianAndSigma(vector)
    return np.abs(vector-m) < (range*s)


def MedianAndSigma(a):
    median = np.median(a)
    MAD = np.median(np.abs(a-median))
    sigma = 1.4826*MAD  # this assumes gauss distribution
    return (median, sigma)


def weigthed_average(y, e=None, axis=0):
    if e is None:
        e = np.ones_like(y)
    if (axis != 0):
        y = y.transpose()
        e = e.transpose()
    (n0, n1) = y.shape
    yout = np.empty(n1)
    eout = np.empty(n1)
    for i in range(n1):
        toav = y[:, i]
        valid = np.isfinite(toav)
        toav = toav[valid]
        w = 1/e[valid, i]**2
        yout[i] = np.sum(toav*w)/np.sum(w)
        eout[i] = np.sqrt(1/np.sum(w))
    return yout, eout

# CONVERSION ###


def convert(value=1, unit="eV", out="nm"):
    if (unit == "eV") & (out == "nm"):
        return 1239.8/value


def deg(rad):
    """ convert radians to deg """
    return np.rad2deg(rad)


def deg(rad):
    """ convert deg to radians"""
    return np.deg2rad(deg)


# LMFIT ####
def fitGaussian(x, y):
    import lmfit
    from lmfit.models import GaussianModel
    mod = GaussianModel()
    pars = GaussianModel().guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    return out.best_fit, out.params


def fitGaussianAndConst(x, y):
    import lmfit
    from lmfit.models import GaussianModel, ConstantModel
    mod = GaussianModel()+ConstantModel()
    pars = GaussianModel().guess(y, x=x)
    pars = pars+ConstantModel().guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    return out.best_fit, out.params


def profile2D(img, pix=4.6e-3, plot=True, fitrange=200, recenter=False, withConstant=False):
    """ find peak in 2D array;
        first index is y direction (row)

        Parameters
        ----------
        pix: float
          the pixel size (can be a tuple/list)

        fitrange: pixels
          to limit the fitting range to +/- the max

        recenter: bool
          recenter plot and fit coordinate to max

        withConstant: bool
          fit with gaussian + constant or gaussian only
    """

    if isinstance(pix, (float, int)):
        pix = (pix, pix)

    # find center
    xmax_idx = np.argmax(img.mean(0))
    ymax_idx = np.argmax(img.mean(1))

    x = np.arange(img.shape[1]) * pix[1]
    y = np.arange(img.shape[0]) * pix[0]
    if recenter:
        x -= pix[1]*xmax_idx
        y -= pix[0]*ymax_idx

    # prepare and perform fit
    idx_fit_x = slice(xmax_idx-fitrange, xmax_idx+fitrange)
    idx_fit_y = slice(ymax_idx-fitrange, ymax_idx+fitrange)

    py = img.mean(1)
    yfit = y[idx_fit_y]
    pyfit = py[idx_fit_y]
    if withConstant:
        fy, parsy = fitGaussianAndConst(yfit, pyfit)
    else:
        fy, parsy = fitGaussian(yfit, pyfit)

    px = img.mean(0)
    xfit = x[idx_fit_x]
    pxfit = px[idx_fit_x]
    if withConstant:
        fx, parsx = fitGaussianAndConst(xfit, pxfit)
    else:
        fx, parsx = fitGaussian(xfit, pxfit)

    if plot:
        fig = plt.figure()
        ax0 = fig.add_axes([0.1, 0.1, 0.5, 0.5])
        c = ax0.pcolormesh(x, y, img, cmap=plt.cm.gray_r)
        c.set_clim(0, img.max())

        ax = fig.add_axes([0.6, 0.1, 0.3, 0.5], sharey=ax0)
        ax.plot(py, y)
        ax.plot(pyfit, yfit, ".")
        ax.plot(fy, yfit)
        ax.set_title("fwhm = %.3f" % parsy['fwhm'].value)

        ax = fig.add_axes([0.1, 0.6, 0.5, 0.3], sharex=ax0)
        ax.plot(x, px)
        ax.plot(xfit, pxfit, ".")
        ax.plot(xfit, fx)
        ax.set_title("fwhm = %.3f" % parsx['fwhm'].value)


# IMINUIT RELATED ###

def iminuitClass(modelFunc):
    """ build a class to help fitting, first argumes of modelFunc should be x"""
    import inspect
    import iminuit
    import iminuit.util
    args = inspect.getargspec(modelFunc).args
    defaults = inspect.getargspec(modelFunc).defaults
    if defaults is not None:
        nDef = len(defaults)
        args = args[:-nDef]
    args_str = ",".join(args[1:])

    class iminuitFit(object):
        def __init__(self, x, data, init_pars, err=1.):
            self.x = x
            self.data = data
            self.err = err
            self.init_pars = init_pars
            self.func_code = iminuit.util.make_func_code(
                args[1:])  # docking off independent variable
            self.func_defaults = None  # this keeps np.vectorize happy

        def __call__(self, *arg):
            return self.chi2(*arg)

        def model(self, *arg):
            return modelFunc(self.x, *arg)

        def chi2(self, *arg):
            c2 = (self.model(*arg)-self.data)/self.err
            return np.sum(c2*c2)

        def fit(self, showInit=True, showPlot=True, doFit=True, doMinos=False):
            import pylab as plt
            p = self.init_pars
            if "errordef" not in p:
                p["errordef"] = 1.
            # m = iminuit.Minuit(self,print_level=0,pedantic=False,**p)
            m = iminuit.Minuit(self, **p)
            if showInit:
                model = self.model(*m.args)
                plt.figure("initial pars")
                plt.grid()
                plt.plot(self.x, self.data, "o")
                plt.plot(self.x, model, 'r-', linewidth=2)
                raw_input()
            if doFit:
                m.migrad()
                if doMinos:
                    m.minos()
                    for p in m.parameters:
                        err = m.get_merrors()[p]
                        err = "+ %.4f - %.4f" % (
                            np.abs(err["upper"]), np.abs(err["lower"]))
                        print("%10s %.4f %s" % (p, m.values[p], err))
                else:
                    for p in m.parameters:
                        err = m.errors[p]
                        err = "+/- %.4f" % (err)
                        print("%10s %.4f %s" % (p, m.values[p], err))

            model = self.model(*m.args)
            if (showPlot):
                plt.figure("final fit")
                plt.grid()
                plt.plot(self.x, self.data, "o")
                plt.plot(self.x, model, 'r-', linewidth=2)
            self.m = m
            return m, self.x, self.data, model

    return iminuitFit


def iminuitParsToStr(iminuit, withErrs=True, withFixed=True):
    values = iminuit.values
    errs = iminuit.errors
    pnames = list(values.keys())
    lenParNames = max([len(p) for p in pnames])
    fmt = "%%%ds" % lenParNames
    pnames.sort()
    res = []
    for p in pnames:
        v = values[p]
        e = errs[p]
        isf = iminuit.is_fixed(p)
        if not withFixed and isf:
            continue
        v, e = approx_err(v, e, asstring=True)
        if isf:
            e = "fixed"
        s = fmt % p
        if withErrs:
            s += " = %s +/- %s" % (v, e)
        else:
            s += " = %s" % (v)
        res.append(s)
    return res


# VARIOUS ###

def myProgressBar(N, title="Percentage"):
    import progressbar as pb
    widgets = [title, pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=N)
    return pbar


def chunk(iterableOrNum, size):
    temp = []
    try:
        n = len(iterableOrNum)
    except TypeError:
        n = iterableOrNum
    nGroups = int(np.ceil(float(n)/size))
    for i in range(nGroups):
        m = i*size
        M = (i+1)*size
        M = min(M, n)
        if (m >= n):
            break
        temp.append(slice(m, M))
    try:
        ret = [iterableOrNum[x] for x in temp]
    except TypeError:
        ret = temp
    return ret


def timeres(*K):
    """ return sqrt(a1**2+a2**2+...) """
    s = 0
    for k in K:
        s += k**2
    return np.sqrt(s)


def approx_err(value, err, asstring=False):
    if (not (np.isfinite(err))):
        err = np.abs(value/1e3)
    if (err != 0):
        ndigerr = -int(np.floor(np.log10(err)))
        if (ndigerr < 1):
            ndigerr = 2
        e = round(err, ndigerr)
        ndigerr = -int(np.floor(np.log10(err)))
        v = round(value, ndigerr)
    else:
        v = value
        e = err
    if (asstring):
        return "%s" % v, "%s" % e
    else:
        return v, e


def linFitOld(A, B):
    """ solve Ax = B, returning x """
    temp = np.dot(A.T, B)
    square = np.dot(A.T, A)
    if (np.asarray(square).ndim == 0):
        inv = 1./square
    else:
        inv = np.linalg.inv(square)
    x = np.dot(inv, temp)
    return x


def linFit(A, B, cond=None):
    """ solve Ax = B, returning x """
    from scipy import linalg
    temp = np.dot(A.T, B)
    square = np.dot(A.T, A)
    if (np.asarray(square).ndim == 0):
        inv = 1./square
    else:
        inv = linalg.pinvh(square, cond=cond)
    x = np.dot(inv, temp)
    return x
    # return np.linalg.lstsq(A,B)[0]


def insertInSortedArray(a, v):
    if v > a.max():
        return a
    idx = np.argmin(a < v)
    # move to the right the values bigger than v
    a[idx+1:] = a[idx:-1]
    a[idx] = v
    return a

# Objects ###


def objToDict(o, recursive=True):
    """ convert a DictWrap to a dictionary (useful for saving); it should work for other objects too """
    if isinstance(o, (dict, collections.OrderedDict)):
        return o
    d = dict()
    for k in o.keys():
        try:
            d[k] = objToDict(getattr(o, k))
        except:
            d[k] = o[k]
    return d


class DictWrap(object):
    """ convert d = dict( a=1,b=np.arange(10),c=dict(d=5,e=dict(f=(3,4)) ))
        to object recursively
        r = DictWrap(d); r.c.e.f = (3,4) """

    def __init__(self, d=None, recursive=True):
        if d is not None:
            if sys.version_info.major == 2:
                for k, v in d.iteritems():
                    if recursive and isinstance(v, (dict, collections.OrderedDict)):
                        v = DictWrap(v, recursive=True)
                    setattr(self, k, v)
            else:
                for k, v in d.items():
                    if recursive and isinstance(v, (dict, collections.OrderedDict)):
                        v = DictWrap(v, recursive=True)
                    setattr(self, k, v)

    def __getitem__(self, x):
        return self.__dict__[x]

    def __setitem__(self, x, v):
        self.__dict__[x] = v

    def _add(self, name, data):
        self.__dict__[name] = data

    def __str__(self):
        return "DictWrap with keys %s" % self.keys()

    def keys(self):
        # list is needed because in py3 keys() returs a generator (and the sorting
        # does noe work)
        k = list(self.__dict__.keys())
        k.sort()
        return k

    def asdict(self):
        """ return a dictionary representation (useful for saving) """
        return objToDict(self)

    def __repr__(self):
        return self.__str__()


def dictToObj(d, recursive=True, cleanNames=True):
    """Return a class that has same attributes/values and
       dictionaries key/value
       DictWrap can do recursively but this function cleans up the names allowing
       easier tab completion
    """
    if not isinstance(d, dict):
        return None
    # define a dummy class
    c = DictWrap()
    for elem in d.keys():
        key = elem
        if cleanNames:
            try:
                int(elem)
                key = "value%s" % elem
            except:
                pass
        if recursive and isinstance(d[elem], dict):
            c.__dict__[key] = dictToObj(d[elem])
        else:
            c.__dict__[key] = d[elem]
    return c


def Hdf5ToObj(h5):
    import h5py
    if isinstance(h5, h5py.File) or isinstance(h5, h5py.Group):
        h5hande = h5
    else:
        h5hande = h5py.File(h5, "r")
    ret = DictWrap()
    for h in h5hande:
        name = h.replace(":", "_")
        name = name.replace(".", "_")
        if not isinstance(h5hande[h], h5py.Dataset):
            ret._add(name, Hdf5ToObj(h5hande[h]))
        else:
            ret._add(name, h5hande[h])
    return ret


def dict2obj(d, recursive=True, cleanNames=True):
    print("DEPRECATED: use dictToObj")
    return dictToObj(d, recursive=recursive, cleanNames=cleanNames)


def fac(n):
    import math
# http://rosettacode.org/wiki/Prime_decomposition#Python

    def step(x): return 1 + x*4 - (x/2)*2
    maxq = long(math.floor(math.sqrt(n)))
    d = 1
    q = n % 2 == 0 and 2 or 3
    while q <= maxq and n % q != 0:
        q = step(d)
        d += 1
    res = []
    if q <= maxq:
        res.extend(fac(n//q))
        res.extend(fac(q))
    else:
        res = [n]
    return res


def define_colors(fname="colorbrewer_all_schemes.json"):
    if not os.path.isfile(fname):
        fname = os.path.dirname(os.path.realpath(__file__)) + "/" + fname
        #print("Cannot find colorbrewer_all_schemes.json can't continue")
        #print(fname)
        return
    if globals()["colors"] is None:
        import json
        ff = open(fname, "r")
        f = json.load(ff)
        ff.close()
        colors = dictToObj(f)
        for t1 in colors.keys():
            for seq in colors[t1].keys():
                for nc in colors[t1][seq].keys():
                    colors[t1][seq][nc] = np.asarray(
                        colors[t1][seq][nc]["Colors"])/255.
        globals()["colors"] = colors


def colors_example_nature(nature, title="colors"):
    import pylab as plt
    x = np.arange(10)
    schemes = nature.keys()
    nSchemes = len(schemes)
    nSequenceMax = max([len(nature[s].keys()) for s in schemes])
    fig, ax = plt.subplots(nSchemes, nSequenceMax,
                           sharex=True, sharey=True, num=title)
    for nScheme in range(nSchemes):
        scheme = nature[schemes[nScheme]]
        sequences = scheme.keys()
        nSequences = len(sequences)
        for i in range(nSequences):
            mycolors = nature[schemes[nScheme]][sequences[i]]
            for j, c in enumerate(mycolors):
                ax[nScheme][i].axhline(j, color=c)
            ax[nScheme][i].set_title("%s %s" % (
                schemes[nScheme], sequences[i]))


def colors_example():
    define_colors()
    for nature in colors.keys():
        colors_example_nature(colors[nature], nature)

from scipy import constants
_electron_rest_mass_energy_GeV = constants.electron_mass*constants.speed_of_light**2/constants.electron_volt/1e9

def electron_energy(beta=0.99):
    gamma = 1/np.sqrt(1-beta**2)
    E = (gamma-1)*_electron_rest_mass_energy_GeV
    return E

try:
    define_colors()
except:
    pass

def fwhm(x,y,normalize=np.max,smooth=0,bkg_points=3):
    """ 
    Find fwhm by finding roots of y-ynorm/2
    ynorm is found y the function normalize (max by default)

    """
    from scipy.interpolate import UnivariateSpline

    if len(x) < 2*bkg_points:
        return np.nan

    x1 = x[:bkg_points].mean()
    y1 = y[:bkg_points].mean()
    x2 = x[-bkg_points:].mean()
    y2 = y[-bkg_points:].mean()

    bkg = y1+(y2-y1)/(x2-x1)*(x-x1)
    y = y-bkg

    norm = normalize(y)
    spline = UnivariateSpline(x, y-norm/2, s=smooth)
    roots = spline.roots()  # find the roots

    if len(roots) == 0:
        fwhm = np.nan
    elif len(roots) == 1:
        fwhm = roots[0]
    else:
        fwhm = roots[-1]-roots[0]
    return fwhm


def theta_to_q(wavelength,theta):
    """ return reciprocal vector, theta is in deg """
    t_half_rad = np.deg2rad(theta)/2
    return 4*np.pi/wavelength*np.sin(t_half_rad)

def q_to_theta(wavelength,q,in_rad=False):
    """ return scattering angle in deg (by default) given q and wavelength """
    theta = 2*np.arcsin(q*wavelength/4/np.pi)
    if not in_rad: theta = np.rad2deg(theta)
    return theta

