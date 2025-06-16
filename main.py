# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:02:06 2017

@author: xavier
"""
# Libraries
import audiotools
import detectors
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time
import os
import sys
import logging
import gc
import configparser
import argparse

#import rpy2.robjects.numpy2ri
#from rpy2.robjects.packages import importr

def HaddockDetector(infile, outdir,templateFile, config):
    max_chunk_sec = 1000
    detections = detectors.Detec()

    # load file
    Rec_ref = audiotools.Sound(infile)
    max_chunk_samp = round(max_chunk_sec*Rec_ref.getSamplingFrequencyHz())

    if max_chunk_samp > Rec_ref.getFileDur_samples():
        chunks = [0, Rec_ref.getFileDur_samples()]
    else:
        chunks = list(range(0,Rec_ref.getFileDur_samples(),max_chunk_samp))
        if chunks[-1] < Rec_ref.getFileDur_samples():
            chunks.append(Rec_ref.getFileDur_samples())
    n_chunks = len(chunks)-1
    for chunk_idx, chunk_start in enumerate(chunks[:-1]):
        print('processing chunk ' + str(chunk_idx+1) + '/' + str(n_chunks))
        chunk_stop = chunks[chunk_idx+1]
        Rec = audiotools.Sound(infile)
        Rec.read(channel=config['channel'], chunk = [chunk_start,chunk_stop])
        Rec.waveform = Rec.waveform - np.mean(Rec.waveform)

        # filter
        #Filt = audiotools.Filter('lowpass', config['LowPassFilter_Hz'], 4)
        Filt = audiotools.Filter('bandpass', [config['BandPassFilter_low_Hz'], config['BandPassFilter_high_Hz']], 4)
        Rec.applyFilter(Filt)

        # Show spectrogram and waveforn
        #plt.figure(1)
        #plt.subplot(2, 1, 1)

        # Detection (kurtosis)
        KurtDetector = detectors.KurtosisDetector(Rec, Kurtframe_sec=config['KurtosisFrame_Sec'], Kurtth=config['KurtosisThreshold'], Kurtdelta_sec=config['KurtosisDelta_Sec'])
        detec = KurtDetector.run()
        #plt.figure(1)
        #plt.subplot(2, 1, 2)
        #KurtDetector.plot(displayDetections=True, unit='sec', newFig=False)

        # Template
        template0 = pd.read_csv(templateFile) # load template
        templateFs = round(1/(template0['TimeSec'].iloc[1]-template0['TimeSec'].iloc[0]))
        template = np.array(template0['Amplitude'].values)
        template = audiotools.normalizeVector(template) # normalize
        templateLen = len(template)
        #template = np.gradient(template, 5)

        # Loop through detections
        T1_sample = np.zeros((1,len(detec)))
        T2_sample = np.zeros((1,len(detec)))
        ConfidenceDTW = np.zeros((1,len(detec)))
        #for i in [10]: # range(0,len(detec)):
        for i in range(0,len(detec)):

            # extract detected signal snippet
            dt = int(round(0.06*Rec.getSamplingFrequencyHz()))
            chunk=[max(int(detec.iloc[i]['startTimeSamp']-dt), 0), min(int(detec.iloc[i]['startTimeSamp']+dt),Rec.getFileDur_samples())]
            snip = Rec.extractWaveformSnippet(chunk)

            # readjust window using cumulated energy (focussing on 90% energy)
            #snip.tightenWavformWindow(EnergyPercentage=83)

            # resample test sequence if needed
            if snip.getSamplingFrequencyHz() != templateFs:
                xorig = np.arange(0,snip.getWaveformDur_sec(),1/snip.getSamplingFrequencyHz())
                yorig = snip.getWaveform()
                if len(xorig) > len(yorig):
                    xorig=xorig[0:len(yorig)]
                xinterp = np.arange(0,snip.getWaveformDur_sec(),1/templateFs)
                testVec = np.interp(xinterp, xorig, yorig, left=None, right=None, period=None)
                # Figure
                #plt.figure()
                #plt.plot(xorig, yorig, '.b')
                #plt.plot(xinterp, testVec, '.r')
                #plt.show
            else:
                testVec = snip.getWaveform()

            # Normalize test signal
            testVec = audiotools.normalizeVector(testVec) # normalize

            # recenter detection based on max energy
            EnWin_samp = int(round(0.015 * templateFs))
            xaxis = np.arange(0, len(testVec))
            En1 = pd.DataFrame({'sig': testVec**2}, index=xaxis)
            EnSmooth1 = np.array(En1.rolling(window=EnWin_samp, center=True).mean().values).flatten()
            idPeak=np.where(EnSmooth1 == np.nanmax(EnSmooth1))
            halfWin = round(templateLen/2)
            t2 = idPeak[0][0] + halfWin
            t1 = idPeak[0][0] - halfWin
            if t1 < 0:
                t1 = 0
                t2 = t1 + templateLen
            elif t2 > len(testVec):
                t2 = len(testVec)
                t1 = t2-templateLen
            # stacks new start and stop times
            T1_sample[0][i] = chunk[0] + round((t1/templateFs)*snip.getSamplingFrequencyHz())
            T2_sample[0][i] = chunk[0] + round((t2/templateFs)*snip.getSamplingFrequencyHz())

            #print(T1_sample[0][i])
            #print(T2_sample[0][i])

            #plt.figure()
            #plt.plot(testVec,'k')
            #plt.plot(EnSmooth1,'r')
            #plt.plot([t1, t1],[-1, 1],'--r',[t2, t2],[-1, 1],'--r')

            testVec = testVec[t1:t2]
            testVec = audiotools.normalizeVector(testVec) # normalize
            #testVec = np.gradient(testVec, 5)

            # DTW  measure
            #ConfidenceDTW[0][i], path = fastdtw(template.T, testVec.T, dist=euclidean, radius=1)
            ConfidenceDTW[0][i], path = fastdtw(template.T, testVec.T, dist=1, radius=1)

            #plt.figure()
            #title = 'Detection #' + str(i) + ' (Distance: ' + str(int(ConfidenceDTW[0][i])) + ')'
            #plt.plot(testVec,'k',template,'r')
            #plt.legend(['Test sequence','Template'])
            #plt.title(title)
            #plt.grid()
            #plt.xlabel('Time (sample)')
            #plt.ylabel('Normalized amplitude')

            # Plot
            #title = 'Detection #' + str(i) + ' (Distance: ' + str(int(ConfidenceDTW[0][i])) + ')'
            #snip.plotWaveform(unit='sec', newfig=True,title=title)
            # Save plot
            #outfilename = str(i) + '.png'
            #plt.savefig(os.path.join(outdir, outfilename), bbox_inches='tight')
            #plt.close()

            # delete objects
            del snip

        # plot histogram
        # plt.figure()
        # plt.hist(ConfidenceDTW.T,np.arange(0,100,5))
        # plt.grid
        # plt.savefig(os.path.join(outdir, 'histogram.png'), bbox_inches='tight')

        # populate detection object
        detections_chunk = detectors.Detec()
        detections_chunk.output['startTimeSec'] = np.matrix.flatten((T1_sample+chunk_start)/Rec.getSamplingFrequencyHz())
        detections_chunk.output['stopTimeSec'] = np.matrix.flatten((T2_sample+chunk_start)/Rec.getSamplingFrequencyHz())
        detections_chunk.output['startTimeSamp'] = np.matrix.flatten(T1_sample+chunk_start)
        detections_chunk.output['stopTimeSamp'] = np.matrix.flatten(T2_sample+chunk_start)
        detections_chunk.output['confidence'] = ConfidenceDTW[0][:]
        detections_chunk.output['fileName'] = Rec.getFileName()
        detections_chunk.output['filePath'] = Rec.getFilePath()
        detections_chunk.output['fileExtension'] = Rec.getFileExtension()
        detections_chunk.output['detectorName'] = 'DTWDetector'
        detections_chunk.output['type'] = 'detec'
        detections_chunk.output['freqMinHz'] = Rec.getFilterParameters()['freqs'][0]
        detections_chunk.output['freqMaxHz'] = Rec.getFilterParameters()['freqs'][1]
        detections_chunk.output['species'] = 'FS'
        detections_chunk.output['call'] = 'PK'
        detections_chunk.output['channel'] = int(Rec.getSelectedChannel())
        detections_chunk.output['fileDurationSec'] = Rec.getFileDur_sec()
        detections_chunk.output['comment'] = detec.confidence
        #detections.output = detections.output.append(detections_chunk.output,ignore_index=True)
        detections.output = pd.concat([detections.output,detections_chunk.output],ignore_index=True)
    # delete detections lower than threshold
    detections.output = detections.output[detections.output.confidence<config['MaxDtwDist']].reset_index(drop=True)
    print(str(len(detections.output)) + ' detections')

    # save as PAMlab annotation
    if len(detections.output) > 0:
        #detections.save2Pamlab(outdir)
        detections.save2Raven(outdir)

    del detec, KurtDetector, Rec, ConfidenceDTW, detections
    #return detections.output


def loadConfigFile(configfile):
    # Loads config  files
    cfg = configparser.ConfigParser()
    cfg.read(configfile)
    channel = int(cfg['AUDIO']['Channel'])
    bandPassFilter_low_Hz = float(cfg['AUDIO']['BandPassFilter_low_Hz'])
    bandPassFilter_high_Hz = float(cfg['AUDIO']['BandPassFilter_high_Hz'])
    KurtosisFrame_Sec = float(cfg['DETECTOR']['KurtosisFrame_Sec'])
    KurtosisThreshold = float(cfg['DETECTOR']['KurtosisThreshold'])
    KurtosisDelta_Sec = float(cfg['DETECTOR']['KurtosisDelta_Sec'])
    MaxDtwDist = float(cfg['CLASSIFICATION']['MaxDtwDist'])
    #if type(LowPassFilter_Hz) != list:
    #    LowPassFilter_Hz = [LowPassFilter_Hz]
    config = ({'channel': channel,
               #'LowPassFilter_Hz': LowPassFilter_Hz,
               'BandPassFilter_low_Hz': bandPassFilter_low_Hz,
               'BandPassFilter_high_Hz': bandPassFilter_high_Hz,
               'KurtosisFrame_Sec': KurtosisFrame_Sec,
               'KurtosisThreshold': KurtosisThreshold,
               'KurtosisDelta_Sec': KurtosisDelta_Sec,
               'MaxDtwDist': MaxDtwDist
               })
    return config


def getInputs():
    # parsing of the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str,
                        help="File or directory to process")
    parser.add_argument("output", type=str,
                        help="Output directory")
    parser.add_argument("cfgfile", type=str,
                        help="Configuration file")
    parser.add_argument("templatefile", type=str,
                        help="Template file for DTW")
    parser.add_argument("-r", "--recursive", action="store_true", default=False,
                        help="Process files in input directory recursively")
    parser.add_argument("-of", "--outputfolder", action="store_true", default=False,
                        help="Outputs results in station/recorder folder structure")
    parser.add_argument("-op", "--pamlab", action="store_true", default=True,
                        help="Outputs results in the PAMlab annotation format")
    parser.add_argument("-or", "--raven", action="store_true", default=False,
                        help="Outputs results in the RAVEN annotation format")
    args = parser.parse_args()
    # find out if input arg is a folder or file
    if os.path.isfile(args.input):
        inputIsFile = True
        inputIsDir = False
        infile = args.input
        indir = []
    elif os.path.isdir(args.input):
        inputIsFile = False
        inputIsDir = True
        infile = []
        indir = args.input
    else:
        raise ValueError('The input folder or file does not exist.')
    # checks that the config file exists
    if os.path.isfile(args.cfgfile):
        config = loadConfigFile(args.cfgfile)
    else:
        raise ValueError('The configuration file does not exist.')
    # checks that the template file exists
    if os.path.isfile(args.templatefile):
        templateFile = args.templatefile
    else:
        raise ValueError('The configuration file does not exist.')
    # Creates output folder if it doesn't exist
    outdir = args.output
    if os.path.isdir(args.output) == False:
        os.mkdir(args.output)
    # options
    recurciveMode = args.recursive

    outopts= {'pamlab':args.pamlab,'raven':args.raven,'outputfolder':args.outputfolder}
    #outopts['pamlab'] = args.pamlab
    #outopts['raven'] = args.raven
    #outopts['outputfolder'] = args.outputfolder

    return inputIsFile, inputIsDir, infile, indir, config, templateFile, outdir, recurciveMode, outopts

def main(args=None):
    """The main routine."""


    if len(sys.argv) == 1:
        templateFile = r'C:\Users\xavier.mouy\Documents\Perso\NOAA\Haddock\config\template_2006.csv'
        outdir = r'C:\Users\xavier.mouy\Documents\Perso\NOAA\Haddock\results'
        infile= r'C:\Users\xavier.mouy\Documents\Perso\NOAA\Haddock\data\671399976.201115190218.wav'
        infile= r'C:\Users\xavier.mouy\Documents\Perso\NOAA\Haddock\data\haddock knock files 31may2006.wav'
        config_file = r'C:\Users\xavier.mouy\Documents\Perso\NOAA\Haddock\config\config.ini'

        config = loadConfigFile(config_file)

        if os.path.isdir(outdir) is False:
            os.makedirs(outdir)
        try:
            det = HaddockDetector(infile, outdir,templateFile, config)
            print('done')
        except BaseException as e:
            print('Failed to do something: ' + str(e))

    else:
        # Parse and load input argument of CLI
        inputIsFile, inputIsDir, infile, indir, config, templateFile, outdir, recurciveMode, outopts = getInputs()
        #print(config)

        indir = sys.argv[1]
        outdir = sys.argv[2]
        #plt.close('all')

        # error logs
        logging.basicConfig(filename=os.path.join(outdir, time.strftime("%Y%m%dT%H%M%S") + '_log.txt'), level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

        file_idx=0
        if recurciveMode:
            for path, subdirs, files in os.walk(indir):
                for name in files:
                    if (name.lower().endswith('.wav')) or (name.lower().endswith('.aif')):
                        print('file', file_idx, '/', str(len(files)))
                        relpath = path[len(indir)+1:] # extract folder structure
                        newoutdir = os.path.join(outdir, relpath) # reproduce same folder structure in output directory
                        if os.path.isdir(newoutdir) is False: # create output folder if doesn't exist
                            os.makedirs(newoutdir)
                        infile = os.path.join(path, name)
                        print(infile)
                        logging.info(infile)
                        start_time = time.time()
                        try:
                            HaddockDetector(infile, newoutdir, templateFile, config)
                            gc.collect() # force garbage collection
                            print("--- %s seconds ---" % (time.time() - start_time))
                            logging.info("--- %s seconds ---" % (time.time() - start_time))
                        except BaseException as e:
                            logging.error(str(e))
                    file_idx+=1
        else:
            files_list = os.listdir(indir)
            for file in files_list:
                if (file.lower().endswith('.wav')) or (file.lower().endswith('.aif')):
                    print('file', file_idx, '/', str(len(files_list)))
                    infile = os.path.join(indir, file)
                    print(infile)
                    logging.info(infile)
                    if os.path.isdir(outdir) is False:
                            os.makedirs(outdir)
                    start_time = time.time()
                    HaddockDetector(infile, outdir, templateFile, config)
                    logging.info("--- %s seconds ---" % (time.time() - start_time))
                    print("--- %s seconds ---" % (time.time() - start_time))
                file_idx+=1

if __name__ == "__main__":
    #  Detec = main()
    config = main()
    print('----------------------------------------------')
    print('Process complete.')
    logging.info('Process complete.')

