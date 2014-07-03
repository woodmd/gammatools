import wx
import numpy as np

from gammatools.core.fits_util import SkyImage, SkyCube

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wxagg import Toolbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

def make_projection_plots_skyimage(im):

    plt.figure()

    im.project(0).plot()
    
    plt.figure()

    im.project(1).plot()
    
def make_plots_skycube(im,delta_bin=None,paxis=None,plots_per_fig=None,
                       smooth=False, **kwargs):

    nbins = im.axis(2).nbins()
    if delta_bin is None: delta_bin = nbins
    
    nplots = nbins/delta_bin

    if plots_per_fig is None: plots_per_fig = min(8,nplots)

    
    nfig = nplots/plots_per_fig

    print nfig

    if plots_per_fig > 4 and plots_per_fig <= 8:
        nx, ny = 4, 2
    elif plots_per_fig <= 4:
        nx, ny = 2, 2


    fig_sx = 4.0*nx
    fig_sy = 4.0*ny
        
    for i in range(nfig):

#        fig, axes = plt.subplots(2,4,figsize=(1.5*10,1.5*5))

        fig = plt.figure(figsize=(fig_sx,fig_sy))
        for j in range(plots_per_fig):

            ibin = i*nfig*delta_bin + j*delta_bin

            if ibin >= nbins: break
            
            print i, j, ibin
        
            h = im.marginalize(2,[ibin,ibin+1])
            if smooth: h = h.smooth(0.25)

            emin = im.axis(2).pix_to_coord(ibin)
            emax = im.axis(2).pix_to_coord(ibin+delta_bin)
            
            subplot = '%i%i%i'%(ny,nx,j+1)

            if paxis is None:            
                h.plot(subplot=subplot,**kwargs)
                ax = h.ax()
                ax.set_title('E = [%.3f %.3f]'%(emin,emax))
            else:
                ax = fig.add_subplot(subplot)
                hp = h.project(paxis)
                hp.plot(ax=ax,**kwargs)
                ax.set_xlim(*hp.axis().lims())
#                ax.set_ylim(0)

    
def make_projection_plots_skycube(im,paxis,delta_bin=2):


    nbins = im.axis(2).nbins()
    nfig = nbins/(8*delta_bin)
    
    for i in range(nfig):

        fig, axes = plt.subplots(2,4,figsize=(1.5*10,1.5*5))
        for j in range(8):

            ibin = i*nfig*delta_bin + j*delta_bin

            if ibin >= nbins: break
            
            print i, j, ibin
        
            h = im.marginalize(2,[ibin,ibin+1])
            emin = im.axis(2).pix_to_coord(ibin)
            emax = im.axis(2).pix_to_coord(ibin+delta_bin)

    
            axes.flat[j].set_title('E = [%.3f %.3f]'%(emin,emax))

            hp = h.project(paxis)
            hp.plot(ax=axes.flat[j])
            axes.flat[j].set_xlim(*hp.axis().lims())
            axes.flat[j].set_ylim(0)
        

class Knob:
    """
    Knob - simple class with a "setKnob" method.  
    A Knob instance is attached to a Param instance, e.g., param.attach(knob)
    Base class is for documentation purposes.
    """
    def setKnob(self, value):
        pass


class Param(object):
    """
    The idea of the "Param" class is that some parameter in the GUI may have
    several knobs that both control it and reflect the parameter's state, e.g.
    a slider, text, and dragging can all change the value of the frequency in
    the waveform of this example.  
    The class allows a cleaner way to update/"feedback" to the other knobs when 
    one is being changed.  Also, this class handles min/max constraints for all
    the knobs.
    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """
    def __init__(self, initialValue=None, minimum=0., maximum=1.):
        self.minimum = minimum
        self.maximum = maximum
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []
        
    def attach(self, knob):
        self.knobs += [knob]

    def setMax(self, maximum):

        self.maximum = maximum
        
    def set(self, value, knob=None):
        self.value = value
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    def value(self):
        return self.value

    def constrain(self, value):
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value


class SliderGroup(object):
    def __init__(self, parent, label, pmin, pmax, fn, pdefault=None):
        self.sliderLabel = wx.StaticText(parent, label=label)
        self.sliderText = wx.TextCtrl(parent, -1, style=wx.TE_PROCESS_ENTER)
        self.slider = wx.Slider(parent, -1)
        self.slider.SetMin(pmin)
        self.slider.SetMax(pmax)

        if pdefault is None: self.set(pmin)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderLabel, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
                  border=2)
        sizer.Add(self.sliderText, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL,
                  border=2)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.sizer = sizer

        self.slider.Bind(wx.EVT_SLIDER, self.sliderHandler)
        self.sliderText.Bind(wx.EVT_TEXT_ENTER, self.sliderTextHandler)

        self.fn = fn

    def init(self,pmin,pmax,pdefault=None):

        self.slider.SetMax(pmin)
        self.slider.SetMax(pmax)
        if pdefault is None: self.set(pmin)
        else: self.set(pdefault)
        
    def disable(self):
        self.slider.Enable(False)
        self.sliderText.Enable(False)
        
    def sliderHandler(self, evt):

        print evt.GetInt()
        
        v = evt.GetInt() 
        self.set(v)
        self.fn(v)
        
    def sliderTextHandler(self, evt):
        v = self.sliderText.GetValue()

        print v
        self.set(v)
        self.fn(v)
        
    def set(self, value):
        self.sliderText.SetValue('%i'%value)
        self.slider.SetValue(value)

class FITSViewerApp(wx.App):

    def __init__(self,im):

        print im
        self._im = im
        wx.App.__init__(self)


    def OnInit(self):

        print 'im: ', self._im
        
        self.frame = FITSViewerFrame(self._im,parent=None,
                                     title="FITS Viewer",
                                     size=(1.5*640, 1.5*480))
        
#        self.frame1 = FITSViewerFrame(parent=None,
#                                      title="FITS Viewer",
#                                      size=(640, 480))
        self.frame.Show()
        return True
        
class FITSViewerFrame(wx.Frame):
    def __init__(self, hdulist,hdu=0,*args, **kwargs):
        wx.Frame.__init__(self, *args,**kwargs)

        self.hdulist = hdulist
        self.hdu = hdu
        self.slice = 0
        self.nbin = 1
        self.window = PlotPanel(self)
        self.slider0 = SliderGroup(self, 'Slice',0,6,
                                   fn=self.update_slice)
        self.slider2 = SliderGroup(self, 'NBins',1,6,
                                   fn=self.update_nbin)
        self.slider1 = SliderGroup(self,'HDU',0,6,
                                   fn=self.update_hdu)

        self.slider1.init(0,len(hdulist)-1)

        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
#        sizer.Add(self.window, 1, wx.EXPAND)
        sizer.Add(self.slider0.sizer, 0,
                  wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer.Add(self.slider2.sizer, 0,
                  wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer.Add(self.slider1.sizer, 0,
                  wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)

        sizer0.Add(sizer,1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)
        sizer0.Add(self.window,2, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL)

        self.SetSizer(sizer0)

        self.update_hdu(self.hdu)
        self.update_slice(self.slice)
        
    def update_hdu(self,value):

        self.hdu = int(value)
        
        hdu = self.hdulist[self.hdu]
        
        if hdu.header['NAXIS'] == 3:
            self._im = SkyCube.createFromHDU(hdu)
            self.slider0.init(0,self._im.axis(2).nbins()-1)
            self.slider2.init(1,self._im.axis(2).nbins())

        else:
            self._im = SkyImage.createFromHDU(hdu)
            self.slider0.init(0,0)
            self.slider0.disable()
            self.slider2.init(0,0)
            self.slider2.disable()

        self.update()
        
    def update_slice(self,value):

        self.slice = int(value)
        self.update()

    def update_nbin(self,value):

        self.nbin = int(value)
        self.update()

    def update(self):

        if isinstance(self._im,SkyCube):
            im = self._im.marginalize(2,
                                      bin_range=[self.slice,
                                                 self.slice+self.nbin])
            im.plot()
        else:
            self._im.plot()
            
        self.window.canvas.draw()
        
    def OnPaint(self, event):

        print 'OnPaint'
        self.window.canvas.draw()
        

class PlotPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self._fig = plt.figure(figsize=(5,4), dpi=75)
        self.canvas = FigureCanvasWxAgg(self, -1, self._fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        #Toolbar(self.canvas) #matplotlib toolbar
        self.toolbar.Realize()
        #self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()


#        self.toolbar.update() # Not sure why this is needed - ADS

    
        
    def update(self, value):
        print 'update ', value
        self.toolbar.update()
        self._fig.canvas.draw_idle()

class FourierDemoWindow(wx.Window, Knob):
    def __init__(self, *args, **kwargs):
        wx.Window.__init__(self, *args, **kwargs)
        self.lines = []
        self.figure = plt.Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.canvas.callbacks.connect('button_press_event', self.mouseDown)
        self.canvas.callbacks.connect('motion_notify_event', self.mouseMotion)
        self.canvas.callbacks.connect('button_release_event', self.mouseUp)
        self.state = ''
        self.mouseInfo = (None, None, None, None)
        self.f0 = Param(2., minimum=0., maximum=6.)
        self.A = Param(1., minimum=0.01, maximum=2.)


        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)

        self.draw()
        
        # Not sure I like having two params attached to the same Knob,
        # but that is what we have here... it works but feels kludgy -
        # although maybe it's not too bad since the knob changes both params
        # at the same time (both f0 and A are affected during a drag)
        self.f0.attach(self)
        self.A.attach(self)
        self.Bind(wx.EVT_SIZE, self.sizeHandler)

        self.add_toolbar()
       
    def sizeHandler(self, *args, **kwargs):
        self.canvas.SetSize(self.GetSize())
        
    def mouseDown(self, evt):
        if self.lines[0] in self.figure.hitlist(evt):
            self.state = 'frequency'
        elif self.lines[1] in self.figure.hitlist(evt):
            self.state = 'time'
        else:
            self.state = ''
        self.mouseInfo = (evt.xdata, evt.ydata, max(self.f0.value, .1), self.A.value)

    def mouseMotion(self, evt):
        if self.state == '':
            return
        x, y = evt.xdata, evt.ydata
        if x is None:  # outside the axes
            return
        x0, y0, f0Init, AInit = self.mouseInfo
        self.A.set(AInit+(AInit*(y-y0)/y0), self)
        if self.state == 'frequency':
            self.f0.set(f0Init+(f0Init*(x-x0)/x0))
        elif self.state == 'time':
            if (x-x0)/x0 != -1.:
                self.f0.set(1./(1./f0Init+(1./f0Init*(x-x0)/x0)))
                    
    def mouseUp(self, evt):
        self.state = ''

    def draw(self):
        if not hasattr(self, 'subplot1'):
            self.subplot1 = self.figure.add_subplot(211)
            self.subplot2 = self.figure.add_subplot(212)
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        color = (1., 0., 0.)
        self.lines += self.subplot1.plot(x1, y1, color=color, linewidth=2)
        self.lines += self.subplot2.plot(x2, y2, color=color, linewidth=2)
        #Set some plot attributes
        self.subplot1.set_title("Click and drag waveforms to change frequency and amplitude", fontsize=12)
        self.subplot1.set_ylabel("Frequency Domain Waveform X(f)", fontsize = 8)
        self.subplot1.set_xlabel("frequency f", fontsize = 8)
        self.subplot2.set_ylabel("Time Domain Waveform x(t)", fontsize = 8)
        self.subplot2.set_xlabel("time t", fontsize = 8)
        self.subplot1.set_xlim([-6, 6])
        self.subplot1.set_ylim([0, 1])
        self.subplot2.set_xlim([-2, 2])
        self.subplot2.set_ylim([-2, 2])
        self.subplot1.text(0.05, .95, r'$X(f) = \mathcal{F}\{x(t)\}$', \
            verticalalignment='top', transform = self.subplot1.transAxes)
        self.subplot2.text(0.05, .95, r'$x(t) = a \cdot \cos(2\pi f_0 t) e^{-\pi t^2}$', \
            verticalalignment='top', transform = self.subplot2.transAxes)

    def compute(self, f0, A):
        f = np.arange(-6., 6., 0.02)
        t = np.arange(-2., 2., 0.01)
        x = A*np.cos(2*np.pi*f0*t)*np.exp(-np.pi*t**2)
        X = A/2*(np.exp(-np.pi*(f-f0)**2) + np.exp(-np.pi*(f+f0)**2))
        return f, X, t, x

    def repaint(self):
        self.canvas.draw()

    def setKnob(self, value):
        # Note, we ignore value arg here and just go by state of the params
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        plt.setp(self.lines[0], xdata=x1, ydata=y1)
        plt.setp(self.lines[1], xdata=x2, ydata=y2)
        self.repaint()

    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()

        print 'Adding toolbar'
        
        if wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()
