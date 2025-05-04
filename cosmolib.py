from pylab import *
import numpy as np
import scipy.integrate
import numpy as np
from matplotlib import *
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
import scipy.stats as st
import emcee
import iminuit
from iminuit.cost import LeastSquares




###############################################################################
############################# Cosmology Functions #############################
###############################################################################
def e_z(z,cosmo):
    omegam=cosmo['omega_M_0']
    omegax=cosmo['omega_lambda_0']
    w0=cosmo['w0']
    h=cosmo['h']
    omegak=1.-omegam-omegax
    omegaxz=omegax*(1+z)**(3+3*w0)
    e_z=np.sqrt(omegak*(1+z)**2+omegaxz+omegam*(1+z)**3)
    return(e_z)

def inv_e_z(z,cosmo):
    return(1./e_z(z,cosmo))

def hz(z,cosmo):
    return(cosmo['h']*e_z(z,cosmo))

### Proper distance in Mpc
def propdist(z,cosmo,zres=0.001,accurate=False):
    ### z range for integration
    zmax=np.max(z)
    if zmax < zres:
        nb=101
    else:
        nb=(zmax/zres+1).astype(int)
    zvals=np.linspace(0.,zmax,nb)
    ### integrate
    cumulative=np.zeros(int(nb))
    cumulative[1:]=scipy.integrate.cumulative_trapezoid(1./e_z(zvals,cosmo),zvals)
    ### interpolation to input z values
    propdist=np.interp(z,zvals,cumulative)
    ### curvature
    omega=cosmo['omega_M_0']+cosmo['omega_lambda_0']
    k=np.abs(1-omega)
    if omega == 1:
        propdist=propdist
    elif omega < 1:
        propdist=np.sinh(np.sqrt(k)*propdist)/np.sqrt(k)
    elif omega > 1:
        propdist=np.sin(np.sqrt(k)*propdist)/np.sqrt(k)
    ### returning
    return(propdist*2.99792458e5/100/cosmo['h'])

### Luminosity distance in Mpc
def lumdist(z,cosmo,zres=0.001,accurate=False):
    return(propdist(z,cosmo,zres=zres,accurate=accurate)*(1+z))

### Angular distance in Mpc
def angdist(z,cosmo,zres=0.001,accurate=False):
    return(propdist(z,cosmo,zres=zres,accurate=accurate)/(1+z))

### SNIa distance modulus
def musn1a(z, cosmo):
    dlum = lumdist(z, cosmo)*1e6
    return(5*np.log10(dlum)-5+5*np.log10(cosmo['h']/0.7))
### Age
def lookback(z,cosmo,zres=0.001):
    ### z range for integration
    zmax=np.max(z)
    if zmax < zres:
        nb=101
    else:
        nb=(zmax/zres+1).astype(int)
    zvals=np.linspace(0.,zmax,nb)
    ### integrate
    cumulative=np.zeros(int(nb))
    cumulative[1:]=scipy.integrate.cumulative_trapezoid(1./e_z(zvals,cosmo)/(1+zvals),zvals)
    ### interpolation to input z values
    age=np.interp(z,zvals,cumulative)
    ### Age in Gyr
    return age/100/cosmo['h'] * (3.26 * 1e6 * 365 * 24 * 3600 *3e5) / (365 * 24 * 3600) / 1e9

### Eisenstein & Hu 1998 modelization of decoupling, sound horizon and so on... avoids calling CAMB but is not very accurate...
def rs(cosmo,zd=1059.25):
    o0=cosmo['omega_M_0']-cosmo['omega_n_0']     ### need to remove omega_neutrino as they are relativistic (massless at high z)
    h=cosmo['h']
    ob=cosmo['omega_b_0']
    theta=2.7255/2.7
    zeq=2.5*1e4*o0*h**2*theta**(-4)
    keq=7.46*0.01*o0*h**2*theta**(-2)
    b1=0.313*(o0*h**2)**(-0.419)*(1+0.607*(o0*h**2)**0.674)
    b2=0.238*(o0*h**2)**0.223
    # This is E&H zdrag
    #zd=1291.*(o0*h**2)**0.251/(1+0.659*(o0*h**2)**0.828)*(1+b1*(ob*h**2)**b2)
    # We use instead the value coming from CAMB and for Planck+WP+Highl Cosmology as suggested by J.Rich (it depends mostly on atomic physics) => zd=1059.25
    req=RR(zeq,ob,h,theta)
    rd=RR(zd*1.,ob,h,theta)
    rs=(2./(3*keq))*np.sqrt(6./req)*np.log((np.sqrt(1+rd)+np.sqrt(rd+req))/(1+np.sqrt(req)))
    return(rs)

def RR(z,ob,h,theta):
    return(31.492*ob*h**2*theta**(-4)*((1+z)/1000)**(-1))

def thetastar(cosmo,zstar=1090.49):
        rsval=rs(cosmo,zd=zstar)
        da=angdist(zstar,cosmo,zres=0.001)
        return rsval/(1+zstar)/da
        
###############################################################################
###############################################################################


###############################################################################
########################## Miscellaneous Functions ############################
###############################################################################
def progress_bar(i,n):
    if n != 1:
        ntot=50
        ndone=ntot*i/(n-1)
        a='\r|'
        for k in range(ndone):
            a += '#'
        for k in range(ntot-ndone):
            a += ' '
        a += '| '+str(int(i*100./(n-1)))+'%'
        sys.stdout.write(a)
        sys.stdout.flush()
        if i == n-1:
            sys.stdout.write(' Done \n')
            sys.stdout.flush()
###############################################################################
###############################################################################
            



###############################################################################
########################## Fitting Class (minuit & MCMC) ################
###############################################################################
class Data:
    def __init__(self, x, y, cov, model, pnames=None):
        self.x = x
        self.y = y
        self.model = model
        self.cov = cov
        if np.prod(np.shape(x)) == np.prod(np.shape(cov)):
            self.diag = True
            self.errors = cov
            self.invcov = np.diag(1./self.errors**2)
        else:
            self.diag = False
            self.errors = 1./np.sqrt(cov)
            self.invcov = np.linalg.inv(cov)
        self.fit = None
        self.fitinfo = None
        self.pnames = pnames
        self.fixedpars = None
        
    def __call__(self, mytheta, extra_args=None, verbose=False):
        if self.fixedpars is not None:
            theta = self.p0.copy()
            theta[self.fitpars] = mytheta
        else:
            theta = mytheta
        # theta = mytheta
        self.modelval = self.model(self.x, theta)

        if verbose:
            print('Pars')
            print(theta)
            print('Y')
            print(np.shape(self.y))
            print(self.yvals[0:10])
            print('Model')
            print(np.shape(self.modelval))
            print(self.modelval[:10])
            print('Diff')
            print(np.shape((self.y - self.modelval)))
            print((self.yvals - self.modelval)[0:10])
            print('Diff x invcov')
            print(np.shape((self.y - self.modelval).T @ self.invcov))
            print(((self.y - self.modelval).T @ self.invcov)[0:10])
        logLLH = - 0.5 * (((self.y - self.modelval).T @ self.invcov) @ (self.y - self.modelval))
        if not np.isfinite(logLLH):
            return -np.inf
        else:
            return logLLH

    def plot(self, nn=1000, color=None, mylabel=None, nostat=False):
        p=errorbar(self.x, self.y, yerr=self.errors, fmt='o', color=color, alpha=1)
        if self.fit is not None:
            xx = np.linspace(np.min(self.x), np.max(self.x), nn)
            plot(xx, self.model(xx, self.fit), color=p[0].get_color(), alpha=1, label=mylabel)
        if mylabel is None:
            if nostat == False:
                legend(title="\n".join(self.fit_info))
        else:
            legend()


    def fit_minuit(self, guess, fixpars = None, limits=None, scan=None, renorm=False, simplex=False, minimizer=LeastSquares):
        ok = np.isfinite(self.x) & (self.errors != 0)

        ### Prepare Minimizer
        if self.diag == True:
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)
        else:
            print('Non diagonal covariance not yet implemented: using only diagonal')
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)

        ### Instanciate the minuit object
        if simplex == False:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames)
        else:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames).simplex()
        
        ### Limits
        if limits is not None:
            mylimits = []
            for k in range(len(guess)):
                mylimits.append((None, None))
            for k in range(len(limits)):
                mylimits[limits[k][0]] = (limits[k][1], limits[k][2])
            m.limits = mylimits

        ### Fixed parameters
        if fixpars is not None:
            for k in range(len(guess)):
                m.fixed["x{}".format(k)]=False
            for k in range(len(fixpars)):
                m.fixed["x{}".format(fixpars[k])]=True

        ### If requested, perform a scan on the parameters
        if scan is not None:
            m.scan(ncall=scan)

        ### Call the minimization
        m.migrad()  

        ### accurately computes uncertainties
        m.hesse()   

        ch2 = m.fval
        ndf = len(self.x[ok]) - m.nfit
        self.fit = m.values

        self.fit_info = [
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ch2:.1f} / {ndf}",
        ]
        for i in range(len(guess)):
            vi = m.values[i]
            ei = m.errors[i]
            self.fit_info.append(f"{m.parameters[i]} = ${vi:.3f} \\pm {ei:.3f}$")

        if renorm:
            m.errors *= 1./np.sqrt(ch2/ndf)

        return m, ch2, ndf

    def run_mcmc(self, p0, allvariables, nbmc=3000, fixpars=None, nwalkers=32, nsigmas=3., fidvalues=None):
        if fidvalues is not None:
            p0 = fidvalues
        if fixpars is not None:
            self.fixedpars = fixpars
            self.p0 = p0
            fitpars = []
            for i in range(len(allvariables)):
                if i not in fixpars:
                    fitpars.append(i)
            self.fitpars = np.array(fitpars)

        print('fixpars',fixpars)
        print('self.fixedpars',self.fixedpars)

        ### Do a minuit fit first
        fitm, ch2, ndf = self.fit_minuit(p0, fixpars=fixpars)
        parm = np.array(fitm.values)
        errm = np.array(fitm.errors)
        print('parm', parm)
        print('errm',errm)
        
        ndim = len(p0)
        pos = np.zeros((nwalkers, ndim))
        for d in range(ndim):
            pos[:, d] = np.random.randn(nwalkers) * np.sqrt(errm[d]) * nsigmas + parm[d]
        print('Ndim init:', ndim)
        if fixpars is not None:
            ndim = len(allvariables) - len(self.fixedpars)
        print('New ndim:', ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__call__)
        if fixpars is not None:
            print('Len(pos):', np.shape(pos))
            print('len(fixepars):', len(fixpars))
            pos = pos[:, self.fitpars]
            print('New len(pos):', np.shape(pos))
        ## Burn
        print('Burning')
        state = sampler.run_mcmc(pos, nbmc//3, progress=True)
        sampler.reset()
        ## sample
        print('Sampling')
        sampler.run_mcmc(state, nbmc, progress=True)

        allchains = sampler.get_chain(flat=True)
        chains = {}
        num = 0
        for i in range(len(allvariables)):
            if fixpars is None:
                chains[allvariables[i]] = allchains[:,i]
            else:
                if i in self.fitpars:
                    chains[allvariables[i]] = allchains[:,num]
                    num += 1
        return chains



class Datas(Data):
    def __init__(self, datalist, pnames=None):
        self.ndatas = len(datalist)
        self.datas = []
        for i in range(len(datalist)):
            self.datas.append(datalist[i])
        self.pnames = pnames
        self.fixedpars = None

    def __call__(self, mytheta, extra_args=None, verbose=False):
        logLLH = 0.
        for i in range(self.ndatas):
            logLLH += self.datas[i](mytheta, extra_args=extra_args, verbose=verbose)
        return logLLH

    def fit_minuit(self, guess, fixpars = None, limits=None, scan=None, renorm=False, simplex=False, minimizer=LeastSquares):
        for i in range(self.ndatas):
            m, ch2, ndf = self.datas[i].fit_minuit(guess, fixpars=fixpars, limits=limits, scan=scan, renorm=renorm, simplex=simplex, minimizer=minimizer)
        return m, ch2, ndf



def thepolynomial(x,pars):
    f=np.poly1d(pars)
    return(f(x))

def do_minuit(x,y,covarin,guess,functname=thepolynomial, verbose=True, fixpars=None):
    data = Data(x,y,covarin, functname)
    if verbose:
        print('Fitting with Minuit')
    fitm, ch2, ndf = data.fit_minuit(guess, fixpars=fixpars)
    if verbose:
        print('Chi2 = {}'.format(ch2))
        print('ndf = {}'.format(ndf))
        print('Fitted values:')
        print(np.array(fitm.values))
        print('Errors:')
        print(np.array(fitm.errors))
        print('Covariance:')
        print(np.array(fitm.covariance))
    return fitm, np.array(fitm.values), np.array(fitm.errors), np.array(fitm.covariance), ch2, ndf

def matrixplot(chain,vars,col,sm,limits=None,nbins=None,doit=None,alpha=0.7,labels=None):
    nplots=len(vars)
    if labels is None: labels = vars
    if doit is None: doit=np.repeat([True],nplots)
    mm=np.zeros(nplots)
    ss=np.zeros(nplots)
    for i in range(nplots):
        if vars[i] in chain.keys():
            mm[i]=np.mean(chain[vars[i]])
            ss[i]=np.std(chain[vars[i]])
    if limits is None:
        limits=[]
        for i in range(nplots):
            limits.append([mm[i]-3*ss[i],mm[i]+3*ss[i]])
    num=0
    for i in range(nplots):
         for j in range(nplots):
            num+=1
            if (i == j):
                a=subplot(nplots,nplots,num)
                a.tick_params(labelsize=8)
                if i == nplots-1: xlabel(labels[j])
                var=vars[j]
                xlim(limits[i])
                ylim(0,1.2)
                if (var in chain.keys()) and (doit[j]==True):
                    if nbins is None: nbins=100
                    bla=np.histogram(chain[var],bins=nbins,density=True)
                    xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
                    yhist=gaussian_filter1d(bla[0],ss[i]/5/(xhist[1]-xhist[0]))
                    plot(xhist,yhist/max(yhist),color=col, label = '{0:.2g} +/- {1:.2g}'.format(np.mean(chain[var]), np.std(chain[var])))
                    legend(loc='upper left',frameon=False,fontsize=8)
            if (i>j):
                a=subplot(nplots,nplots,num)
                a.tick_params(labelsize=8)
                var0=labels[j]
                var1=labels[i]
                xlim(limits[j])
                ylim(limits[i])
                if i == nplots-1: xlabel(var0)
                if j == 0: ylabel(var1)
                if (vars[i] in chain.keys()) and (vars[j] in chain.keys()) and (doit[j]==True) and (doit[i]==True):
                    a0=cont(chain[vars[j]],chain[vars[i]],color=col,nsmooth=sm,alpha=alpha)
    return(a0)
    
def getcols(color):
    if color == 'blue':
        cols=['SkyBlue','MediumBlue']
    elif color == 'red':
        cols=['LightCoral','Red']
    elif color == 'green':
        cols=['LightGreen','Green']
    elif color == 'pink':
        cols=['LightPink','HotPink']
    elif color == 'orange':
        cols=['Coral','OrangeRed']
    elif color == 'yellow':
        cols=['Yellow','Gold']
    elif color == 'purple':
        cols=['Violet','DarkViolet']
    elif color == 'brown':
        cols=['BurlyWood','SaddleBrown']
    return(cols)


def cont(x,y,xlim=None,ylim=None,levels=[0.9545,0.6827],alpha=0.7,color='blue',nbins=256,nsmooth=4,Fill=True,**kwargs):
    levels.sort()
    levels.reverse()
    cols=getcols(color)
    dx=np.max(x)-np.min(x)
    dy=np.max(y)-np.min(y)
    if xlim is None: xlim=[np.min(x)-dx/3,np.max(x)+dx/3]
    if ylim is None: ylim=[np.min(y)-dy/3,np.max(y)+dy/3]
    range=[xlim,ylim]
    a,xmap,ymap=np.histogram2d(x,y,bins=256,range=range)
    a=np.transpose(a)
    xmap=xmap[:-1]
    ymap=ymap[:-1]
    dx=xmap[1]-xmap[0]
    dy=ymap[1]-ymap[0]
    z=scipy.ndimage.filters.gaussian_filter(a,nsmooth)
    z=z/np.sum(z)/dx/dy
    sz=np.sort(z.flatten())[::-1]
    cumsz=integrate.cumulative_trapezoid(sz)
    cumsz=cumsz/max(cumsz)
    f=interpolate.interp1d(cumsz,np.arange(np.size(cumsz)))
    indices=f(levels).astype('int')
    vals=sz[indices].tolist()
    vals.append(np.max(sz))
    vals.sort()
    if Fill:
        for i in np.arange(np.size(levels)):
            contourf(xmap, ymap, z, vals[i:i+2],colors=cols[i],alpha=alpha,**kwargs)
    else:
        contour(xmap, ymap, z, vals[0:1],colors=cols[1],**kwargs)
        contour(xmap, ymap, z, vals[1:2],colors=cols[1],**kwargs)
    a=Rectangle((np.max(xmap),np.max(ymap)),0.1,0.1,fc=cols[1])
    return(a)

   
###############################################################################
###############################################################################
            
