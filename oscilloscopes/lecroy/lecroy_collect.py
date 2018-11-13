import datastorage
import time
import numpy as np
import progressbar

def take_data(Npulse=10):
  import lecroy
  scope=lecroy.LeCroyScope("10.0.0.1")
  Nsamples = scope.get_waveform(2)[1].shape[0]
  ch2 = np.zeros( (Npulse,Nsamples) )
  t0 = time.time()
  p = progressbar.ProgressBar(Npulse)
  p.start()
  for i in range(Npulse):
    scope.trigger()
    ch2[i] = scope.get_waveform(2)[1]
#    ch4[i] = scope.get_waveform(4)[1]
    p.update(i)
    #print((i+1)/(time.time()-t0))
  p.finish()
  return datastorage.DataStorage( t = scope.get_xaxis(), ch2 = ch2)

def ana(fname):
  data=datastorage.DataStorage(fname)
  ch3 = data.ch3[:,12000:24000]
  ch3_base = ch3[:,:2000].mean(1)
  ch3 = ch3 - ch3_base[:,np.newaxis]
  ch4 = data.ch4[:,12000:24000]
  ch4_base = ch4[:,:2000].mean(1)
  ch4 = ch4 - ch4_base[:,np.newaxis]
  ch3_sum = ch3.sum(axis=1)
  ch4_sum = ch4.sum(axis=1)
  return ch3_sum,ch4_sum
  
