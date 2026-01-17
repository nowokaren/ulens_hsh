from __future__ import print_function
import os
import io
import time
import imexam
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import wcs
from pathlib import Path
from astropy.io import fits
from scipy.spatial import distance
from datetime import datetime as dt
#from photutils import DAOStarFinder #IRAFStarFinder
from imexam.math_helper import gfwhm
from reproject import reproject_interp
from imexam.imexamine import Imexamine
from ccdproc import ImageFileCollection
from astropy.visualization import simple_norm
from photutils import ModeEstimatorBackground
from reproject.mosaicking import reproject_and_coadd
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils import aperture_photometry,CircularAperture,CircularAnnulus

from aligRAF import center_e,geomap_e,geotran_e,imcombine_e
###############################################################################
# Auxiliar functions
###############################################################################
#Delete files from possible previous run from the list
def remove(filt):
    os.system('rm -f *'+filt.lower()+'*_trans0.fits')
    os.system('rm -f *'+filt.lower()+'*_trans.fits')
    os.system('rm -f *'+filt.lower()+'*_scaled.fits')
    os.system('rm -f *'+filt.lower()+'_comb.fits')
    os.system('rm -f *'+filt.lower()+'_comb2.fits')
    print()
    return()

#Show images in ds9
def showinds9(img, mark_region = 'no', positions = []):
    viewer = imexam.connect("alds9")
    viewer.frame(1)
    viewer.load_fits(img) 
    viewer.zoomtofit()
    viewer.scale()
    if mark_region == 'yes':
        if len(positions) == 0:
            print('Position can not be empty')
        else:
            viewer.mark_region_from_array(positions,size=15) 
    viewer.close()
    return()

#Check if a given pair of XY coordinates are in the image.
def is_in_image(x, y):
    out = bool
    if x > 10 and x < 1014 and y > 10 and y < 1014:
        out = True
    else:
        out = False
    return out

#Convert RADEC[deg] to XY
def radec2xy(img,stars):
    hdulist   = fits.open(img)
    w         = wcs.WCS(hdulist[0].header)
    wcs_coord = np.array(stars[[0,1]])
    pix_coord = w.wcs_world2pix(wcs_coord,0)
    xy=[]
    for i in range(len(pix_coord)):
        if is_in_image(pix_coord[i][0],pix_coord[i][1]):
            xy.append((pix_coord[i][0],pix_coord[i][1]))
        else:
            print(f'WARNING: Star {i + 1} is not in image {img}')
    return(xy)

#Calculate mean FWHM in X and Y axis for each image.
def FWHM_im(filename,xy):
    im  = Imexamine()
    #Calculate FWHM
    print(filename+' centers:')
    print()
    data = fits.getdata(filename)
    im.set_data(data)
    measures = {} #Save all the parameters
    mfwhmy   = [] #Save mean fwhm(y) 
    mfwhmx   = [] #Save mean fwhm(x)
    xycntr   = [] #Save coordinates
    xycntr_err = [] #Save coordinates errors
    #For each position in position array
    for x,y in xy:
        #Gaussian fit
        amp, xcntr, ycntr, xsigma, ysigma = im.gauss_center(x,y)
        #fwhm(x), fwhm(y) calculation
        fwhmx, fwhmy    = gfwhm(xsigma, ysigma) 
        measures[(x,y)] = [xcntr, ycntr, xsigma, ysigma, fwhmx, fwhmy] 
        xycntr.append([xcntr,ycntr]) 
        xycntr_err.append([xsigma,ysigma]) 
        mfwhmy.append(fwhmy)
        mfwhmx.append(fwhmx)
    #Show in ds9 the XY position of the stars in img
    print()
    print('Reference stars marked in ds9 for image: '+filename+'.\n'+\
          'Centers of stars in image:')
    print()
    showinds9(filename,mark_region='yes',positions=measures)
    fig, ax = plt.subplots(figsize=(6, 6))
    norm_img = simple_norm(data, 'power',power=3.,percent=99.)
    ax.imshow(data,norm=norm_img,origin='lower',cmap='gray')
    for r in range(len(xycntr)):
        circle=plt.Circle((xycntr[r][0],xycntr[r][1]),
                          15,color='r',fill=False)
        ax.add_artist(circle)
    print('Reference stars positions saved in  '+filename+'_refstars.jpg file')
    if not os.path.exists(objname+'_refstars'):
        os.system("mkdir %s"%objname+'_refstars')
    plt.savefig(str(mpath)+'/'+objname+'_refstars'+'/'+filename+'_refstars.jpg')
    print()
    print("%.50s %.11s %.2f %.11s %.2f %.11s %.2f %.11s %.2f \n" %
          (filename+' fwhm info:\n','mean_fwhmx = ',np.mean(mfwhmx),\
           'std_fwhmx = ',np.std(mfwhmx),'mean_fwhmy = ',\
           np.mean(mfwhmy),'std_fwhmy = ',np.std(mfwhmy)))
    print()
    time.sleep(0.5)
    return(np.mean(mfwhmx),np.std(mfwhmx),np.mean(mfwhmy),np.std(mfwhmy))

#Interact with images to select stars in reference image in case of no WCS
def select_xy(img):
    out_file=img+'.coo'
    #Remove auxiliary files
    os.system('rm -f fwhm1.imx')
    os.system('rm -f fwhm11.imx')
    print('#'*45)
    print('Interaction with ds9 started.')
    print('Use "a" to select reference stars: ')
    print('#'*45)
    print()
    #time.sleep(1.5)
    #Connect to ds9 in order to select stars
    viewer=imexam.connect("alds9") 
    viewer.frame(1) 
    viewer.load_fits(img)
    viewer.zoomtofit() 
    viewer.scale()
    #Save the same data that appears on screen
    viewer.setlog(filename='fwhm1.imx')
    #Define radius where to calculate the position
    viewer.set_plot_pars('a','radius',10)
    viewer.imexam()
    #Close log
    viewer.setlog(on=False) 
    viewer.close()
    #Copy log file to erase empty lines    
    with open(str(mpath)+'/fwhm1.imx') as infile,\
         open('fwhm11.imx', 'w') as outfile:
        for line in infile:
            if not line.strip(): continue  # skip the empty line
            outfile.write(line)  # non-empty line. Write it to output
    #Read x,y, fwhmx & fwhmy of the selected stars
    log_f=np.loadtxt('fwhm11.imx',dtype='str',delimiter='\t',usecols=(0,))
    x=[]
    y=[]
    fwhmx=[]
    fwhmy=[]
    for j in range(2,len(log_f),3):
        x.append(float(log_f[j][:6]))
        y.append(float(log_f[j][6:30]))
        fwhmx.append(float(log_f[j][-9:-5]))
        fwhmy.append(float(log_f[j][-4:]))   
    xy=[]
    for i in range(len(x)):
        xy.append((x[i],y[i]))
    #Show the position of the selected stars on ds9
    print(img+' stars marked in ds9.\nCenters of given stars in image:')
    showinds9(img, mark_region = 'yes', positions = xy)
    #Write positions in file
    xy_w= str(np.array(xy))
    xy_w=xy_w.replace("["," ")
    xy_w=xy_w.replace("]"," ")
    aux=open(out_file,'w+')
    aux.write(xy_w)
    aux.close()
    print(out_file+' file created with the xy position of the given stars')
    return(out_file)

#Calculate backgroud value and mean flux for each (bkg subtracted) image
def bg_flux_im(filename,xy,fwhm):
    data = fits.getdata(filename)
    #Image background
    sigma_clip = SigmaClip(sigma=3.)
    bkg = ModeEstimatorBackground(sigma_clip=sigma_clip)
    bkg_value  = bkg.calc_background(data)
    #Subtract backgrounf of each image
    data_b     = data-bkg_value
    #Aperture photometry
    aperture_r = fwhm
    aperture   = CircularAperture(xy,r=aperture_r)
    phot_table = aperture_photometry(data_b, aperture)
    annulus_r_in  = 10.
    annulus_r_out = 20.
    annulus_aperture = CircularAnnulus(xy, r_in=annulus_r_in,
                                       r_out=annulus_r_out)
    apers = [aperture, annulus_aperture]
    phot_table_bg = aperture_photometry(data_b, apers)
    bkg_mean = phot_table_bg['aperture_sum_1'] / annulus_aperture.area 
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table_bg['aperture_sum_0'] - bkg_sum
    phot_table_bg['final_sum'] = final_sum
    #Image mean flux value 
    flux_value=np.nanmean(np.array(phot_table_bg['final_sum']))
    return(flux_value,bkg_value)

###############################################################################
# Alineation and combination functions
###############################################################################
#Alingment without wcs
def shift(imfiles):
    #Read image headers
    hdu_list = [fits.open(imfiles[0][i])[0] for i in range(len(imfiles))]
    #Save final files
    final_files=[]
    #Selecte reference image
    if len(hdu_list)>1:
        ref_index=int(len(hdu_list)/2)
        ref_img=imfiles[0][ref_index]
    else:
        ref_index=0
        ref_img=imfiles[0][ref_index]
    print()
    print('#'*45)
    counter=len(imfiles)
    print('Number of images to interact with: ')
    print(counter)
    print()        
    #Select coordinates in reference image 
    coordfile1=select_xy(ref_img)
    #Center reference image
    out_cen1=center_e(str(mpath.parents[2])+'/scripts/',ref_img,coordfile1)
    #Create input for geomap
    skip=np.arange(0,500,1)
    xyc1=pd.read_csv(out_cen1,delim_whitespace=True,comment='#',
                     header=None,skiprows=skip[0::2])
    time.sleep(0.5)
    for i in range(len(imfiles[0])):
        if i!=ref_index:
            print('#'*45)
            counter-=1
            print('Remaining images to interact with ',counter)
            #Select coordinates
            coordfile2=select_xy(imfiles[0][i])
            print('Coordinate files: ')
            print(coordfile1)
            print(coordfile2)
            #Center    
            out_cen2=center_e(str(mpath.parents[2])+'/scripts/',
                              imfiles[0][i],coordfile2)
            #Create input file for geomap  
            xyc2=pd.read_csv(out_cen2,delim_whitespace=True,comment='#',
                             header=None,skiprows=skip[0::2])
            xyc=pd.DataFrame([xyc1[0],xyc1[1],xyc2[0],xyc2[1]]).transpose()
            xyc_w= str(np.array(xyc))
            xyc_w=xyc_w.replace("["," ")
            xyc_w=xyc_w.replace("]"," ")
            aux=open('geomap.xy','w+')
            aux.write(xyc_w)
            aux.close()
            #Geomap      
            geomap_e(str(mpath.parents[2])+'/scripts/','geomap.xy','out.geo')
            #Geotran
            geotran_e(str(mpath.parents[2])+'/scripts/',imfiles[0][i],
                      imfiles[0][i][:-5]+'_trans.fits','out.geo','geomap.xy')
            final_files.append(imfiles[0][i][:-5]+'_trans.fits')
            print('Transformed image = '+imfiles[0][i][:-5]+'_trans0.fits')
            print()
        elif i==ref_index:
            final_files.append(imfiles[0][i])
    return(final_files)

#Alingment with some wcs
def algn_with_some_wcs(imfiles,crv):
    final_files=[]
    #Make a list of the images with WCS
    wcs_img=[imfiles[0][j] for j in range(len(crv)) if crv[j] != 'INDEF']
    print()
    print('#'*45)
    counter=len(imfiles)-len(wcs_img)
    print('Number of images to interact with: ')
    print(counter)
    print()
    #Select reference image
    ref_img=wcs_img[0]
    ref_hdu=[fits.open(wcs_img[0])[0]]
    #Select coordinates in reference image 
    coordfile1=select_xy(ref_img)
    #Center reference image
    out_cen1=center_e(str(mpath.parents[2])+'/scripts/',ref_img,coordfile1)
    #Create input for geomap
    skip=np.arange(0,500,1)
    xyc1=pd.read_csv(out_cen1,delim_whitespace=True,comment='#',
                     header=None,skiprows=skip[0::2])
    #Save WCS info to edit shifted images header
    #cv1_ref=hdu[0].header['CRVAL1'])
    time.sleep(0.5)
    for i in range(len(imfiles[0])):
        if imfiles[0][i] not in wcs_img:
            print('#'*45)
            counter-=1
            print('Remaining images to interact with ',counter)
            #Select coordinates
            coordfile2=select_xy(imfiles[0][i])
            print('Coordinate files: ')
            print(coordfile1)
            print(coordfile2)
            #Center    
            out_cen2=center_e(str(mpath.parents[2])+'/scripts/',imfiles[0][i],
                              coordfile2)
            #Create input file for geomap  
            xyc2=pd.read_csv(out_cen2,delim_whitespace=True,comment='#',
                             header=None,skiprows=skip[0::2])
            xyc=pd.DataFrame([xyc1[0],xyc1[1],xyc2[0],xyc2[1]]).transpose()
            xyc_w= str(np.array(xyc))
            xyc_w=xyc_w.replace("["," ")
            xyc_w=xyc_w.replace("]"," ")
            aux=open('geomap.xy','w+')
            aux.write(xyc_w)
            aux.close()
            #Geomap      
            geomap_e(str(mpath.parents[2])+'/scripts/','geomap.xy','out.geo')
            #Geotran
            geotran_e(str(mpath.parents[2])+'/scripts/',imfiles[0][i],
                      imfiles[0][i][:-5]+'_trans0.fits','out.geo','geomap.xy')
            #Edit header of shifted images
            orgnl_img=[fits.open(imfiles[0][i])[0]]
            shift_img=[fits.open(imfiles[0][i][:-5]+'_trans0.fits')[0]]
            new_data = shift_img[0].data
            new_head=orgnl_img[0].header[:-3]+ref_hdu[0].header[45:]
            fits.writeto(imfiles[0][i][:-5]+'_trans0.fits', new_data,
                         new_head,overwrite=True)
            final_files.append(imfiles[0][i][:-5]+'_trans0.fits')
            print('Transformed image = '+imfiles[0][i][:-5]+'_trans0.fits')
            print()
    return(final_files+wcs_img)

#Alingment with wcs
def algn_with_wcs(imfiles,band):
    #Read image headers
    hdu_list = [fits.open(imfiles[i])[0] for i in range(len(imfiles))]
    #Use middle image to align
    if len(hdu_list)>1:
        ref_index=int(len(hdu_list)/2)
        hdu1 = hdu_list[ref_index]
    else:
        ref_index=0
        hdu1=hdu_list[ref_index]
    #Check image EXPTIME remove if B exptime is != 90 or other is != 60 
    ETIME = [i.header['EXPTIME'] for i in hdu_list]
    etrem_indx = []
    for i in range(len(ETIME)):
        if band == 'B':
            if ETIME[i] != 90.:
                etrem_indx.append(i)
                print('Image ',imfiles[i])
                print('will be removed due to bad EXPTIME')
        else:
            if ETIME[i] != 60.:
                etrem_indx.append(i)
                print('Image ',imfiles[i])
                print('will be removed due to bad EXPTIME')
    #Reproject ~ align based on WCS solution
    for i in range(len(imfiles)):
        if i != ref_index and i not in etrem_indx:
            hdu2 = hdu_list[i]
            array,footprint = reproject_interp(input_data=hdu2,
                                              output_projection=hdu1.header)
            #Save aligned images into fits files
            fits.writeto(imfiles[i][:-5]+'_trans.fits', array, hdu1.header,
                         overwrite=True)
    #Create and save the combination list with the aligned images
    aligned_imgs = [imfiles[i][:-5]+'_trans.fits'
                    for i in range(len(imfiles)) if i not in etrem_indx
                    and imfiles[i]!=imfiles[ref_index]]
    aligned_imgs.insert(ref_index,imfiles[ref_index])
    return(aligned_imgs)

#Combination
def comb(comb_list,scaled_list):
    #Read the headers of the original images
    hdu_in = [fits.open(comb_list[i])[0] for i in range(len(comb_list))]
    #Name of output file = obj + filt + date
    #obj and filt from first image header.
    objn = comb_list[0].split('_')
    obj  = objn[0][2:] 
    filt = hdu_in[0].header['FILTERS']
    salida = obj+'_'+filt.lower()+'_comb.fits'
    #Combine with IRAF
    imcombine_e(str(mpath.parents[2])+'/scripts/',scaled_list,salida)
    #Calculate mean DATE-OBS
    DOB = pd.Series([dt.strptime(i.header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S.%f")
                     for i in hdu_in])
    mean_DOB = DOB.mean()
    #Calculate mean TIME-OBS = UT and ST
    TOB = pd.Series([dt.strptime(i.header['TIME-OBS'],"%H:%M:%S.%f")
                    for i in hdu_in])
    SidT = pd.Series([dt.strptime(i.header['ST'],"%H:%M:%S.%f")
                    for i in hdu_in])
    mean_TOB  = TOB.mean()
    mean_SidT = SidT.mean()
    #Calculate mean MJD-OBS
    MJD = [i.header['MJD-OBS'] for i in hdu_in]
    mean_MJD = np.mean(MJD)
    #Calculate mean AIRMASS
    AIRM = [i.header['AIRMASS'] for i in hdu_in]
    mean_AIRM = np.mean(AIRM)
    #Read SN coordinates
    sncatcoord=pd.read_csv(str(mpath.parents[1])+'/objetos/'+obj[2:]+'/'+obj[2:]\
                           +'_sn.coo',header=None,delim_whitespace=True,
                           comment='#')
    snradeg  = str(float(sncatcoord[0]))
    sndecdeg = str(float(sncatcoord[1]))
    #Update combined image header
    with fits.open(salida,'update') as f:
        for hdu in f:
            hdu.header['DATE-OBS'] = (mean_DOB.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                                      'mean date of observation (yyy-mm-dd)')
            hdu.header['TIME-OBS'] = (mean_TOB.strftime("%H:%M:%S.%f"),
                                      'mean time at start of observation')
            hdu.header['UT']       = (mean_TOB.strftime("%H:%M:%S.%f"),
                                      'mean universal time')
            hdu.header['ST']       = (mean_SidT.strftime("%H:%M:%S.%f"),
                                      'mean sidereal time')
            hdu.header['MJD-OBS']  = (mean_MJD, 'mean MJD of observation')
            hdu.header['AIRMASS']  = (mean_AIRM, 'mean airmass')
    hdusal=fits.open(salida,'update')
    hdusal[0].header.insert(26,('SNRA',snradeg,'SN RA in Deg'))
    hdusal[0].header.insert(27,('SNDEC',sndecdeg,'SN DEC in Deg'))
    hdusal.close()
    return(salida)

###############################################################################
# Main
###############################################################################
#Select all images taken with same filter
print()
parser=argparse.ArgumentParser()
parser.add_argument('-F',help='filter')
arguments = parser.parse_args()
filt=arguments.F
print('Combining filter ',filt)
print()
#Check active ds9 windows
print('ds9 active sessions: ')
chkds9  = imexam.list_active_ds9() 
nameds9 = ([c[0] for c in chkds9.values()])
#If our ds9 window (alds9) is already open do nothing, else open it
if 'alds9' not in nameds9:
    subprocess.Popen(['ds9', '-title', 'alds9'])
    time.sleep(1.5)
remove(filt)
time.sleep(0.5)
while True:
    print('Selecting files')
    #Select all images taken with same filter
    mpath  = Path().absolute()
    images = ImageFileCollection(str(mpath)+'/',keywords='*')
    lista  = images.files_filtered(imagetyp='object',calibz='subtracted bias',
                                   calibf='flat correction',astromet='yes',
                                   filters=filt)
    imfiles = pd.DataFrame(lista)
    imfiles.columns = [0]
    if not any('trans' in imf for imf in imfiles[0]):
        break

print("Collected images:")
print(*('\t'+im+'\n' for im in imfiles[0]))
print()

#Check if every image has solved wcs a succesfull astrometry
crv=[]
for imf in imfiles[0]:
    #Read image headers
    hdu = [fits.open(imf)[0]]
    #Save crval value
    crv.append(hdu[0].header['CRVAL1'])

if all(cv=='INDEF' for cv in crv):
#Align and combine manually without WCS
    #Shift images
    shift_imgs=shift(imfiles)
    #Append 0 to beggining of image list to know it does not have WCS
    shift_imgs.insert(0,'0')
elif any(cv=='INDEF' for cv in crv) and not all(cv=='INDEF' for cv in crv):
#Align the ones withuot WCS w.r.t. the first image with WCS
#Copy the WCS solution to the mentioned images
#Then combined all images as if the astrometric solution was good for all
    #Shift images
    shift_imgs=algn_with_some_wcs(imfiles,crv)
    #Append 1 to beggining of image list to know it does have WCS
    shift_imgs.insert(0,'1')
else:
    shift_imgs=imfiles[0].tolist()
    #Append 1 to beggining of image list to know it does have WCS
    shift_imgs.insert(0,'1')
    
#Read field stars file
objname=imfiles[0][0].split('_')[0][4:]

#If images have WCS
if shift_imgs[0]=='1':
    #Read reference stars
    stars = pd.read_csv(str(mpath.parents[1])+'/objetos/'+objname+'/'+\
                    objname+'_refpsf.coo', delim_whitespace=True,comment='#',
                    header=None)
    #Get the position xy and measure FWHM for field stars
    #Get the mean FWHM for each image
    xy = []
    im_mean_fwhmx = []
    im_std_fwhmx  = []
    im_mean_fwhmy = []
    im_std_fwhmy  = []
    
    for im in range(1,len(shift_imgs)):
        star_positions=radec2xy(shift_imgs[im],stars)
        xy.append(star_positions)
    
    imfiles1=[]
    
    #Add first image to the list of images to combine.
    #Measure FWHM for field stars
    #Get the mean FWHM for the image
    imfiles1.append(shift_imgs[1])
    fwhm_info = FWHM_im(shift_imgs[1],xy[0])
    im_mean_fwhmx.append(fwhm_info[0])
    im_std_fwhmx.append(fwhm_info[1])
    im_mean_fwhmy.append(fwhm_info[2])
    im_std_fwhmy.append(fwhm_info[3])
    
    #Check if distance is less than 30" 
    #for the first three reference stars in each image
    #Measure FWHM for field stars
    #Get the mean FWHM for each image

    for im in range(2,len(shift_imgs)):
        if distance.euclidean(xy[im-2][0],xy[im-1][0])<=60.\
           and distance.euclidean(xy[im-2][1],xy[im-1][1])<=60.\
           and distance.euclidean(xy[im-2][2],xy[im-1][2])<=60.:
            imfiles1.append(shift_imgs[im])
            fwhm_info=FWHM_im(shift_imgs[im],xy[im-1])
            im_mean_fwhmx.append(fwhm_info[0])
            im_std_fwhmx.append(fwhm_info[1])
            im_mean_fwhmy.append(fwhm_info[2])
            im_std_fwhmy.append(fwhm_info[3])
        else:
            print('*'*75)
            print('Image ',shift_imgs[im],' removed due to big offset')
            print()
    #Get the mean FWHM of the set of images (global mean FWHM)
    glob_mean_fwhmx = np.mean(im_mean_fwhmx) 
    glob_std_fwhmx  = np.std(im_mean_fwhmx)  
    glob_mean_fwhmy = np.mean(im_mean_fwhmy) 
    glob_std_fwhmy  = np.std(im_mean_fwhmy)
    #Remove images with im_FWHM > global_FWHM + 3sigma,
    #save good images in imfiles2
    #Save FWHM to calculate aperture to measure mean flux of each image
    imfiles2=[]
    fwhm_4ap=[]
    for i in range(len(imfiles1)):
        if np.abs(im_mean_fwhmx[i] - glob_mean_fwhmx) <= 3*glob_std_fwhmx \
        and np.abs(im_mean_fwhmy[i] - glob_mean_fwhmy) <= 3*glob_std_fwhmy:
            imfiles2.append(imfiles1[i])
            fwhm_4ap.append(max(im_mean_fwhmx[i],im_mean_fwhmy[i]))

            imfiles2.append(imfiles1[i])
            fwhm_4ap.append(max(im_mean_fwhmx[i-1],im_mean_fwhmy[i-1]))
        else:
            print('*'*75)
            print('Image ', imfiles1[i],' removed due to bad seeing')
            print()
    #Measure background and calculate aperture photometry for
    #selected stars
    #im_meax* values is the mean flux and bkg value per image
    #glob* values are the mead flux and bkg value of all the images
    im_mean_flux = []
    im_mean_bkg  = []
    for i in range(len(imfiles2)):
        fx,bg = bg_flux_im(imfiles2[i],xy[i],fwhm_4ap[i])
        im_mean_flux.append(fx)
        im_mean_bkg.append(bg)
    glob_mean_flux = np.mean(im_mean_flux)
    #Calculate reference flux and background to subtract to every image
    ref_img = imfiles2[int((len(imfiles2)/2)-1)]
    ref_flux,ref_bkg_value = bg_flux_im(ref_img,xy[int((len(imfiles2)/2)-1)],
                                        fwhm_4ap[int((len(imfiles2)/2)-1)])
    #Align images
    trans_files = algn_with_wcs(imfiles2,filt)
    #Scale images by bkg and flux and save them
    scal_files = []
    for i in range(len(trans_files)):
        hdu  = fits.open(trans_files[i])[0]
        data = fits.getdata(trans_files[i])
        #Scaled image: (img-bkg)*(ref_flux/img_flux)
        data_norm = (data-im_mean_bkg[i])*(ref_flux/im_mean_flux[i])
        simgname  = trans_files[i][:-5]+'_scaled.fits'
        scal_files.append(simgname)
        fits.writeto(simgname,data_norm,hdu.header,overwrite=True)
    #Save scaled aligned images to list to then combine them
    with open('input_combine.lst','w+') as outfile:
        outfile.write('\n'.join(scal_files))
        outfile.write('\n')
    #Combine images 
    comb_img = comb(imfiles2,'input_combine.lst')
    #Add background to combined image
    hdu_comb    = fits.open(comb_img)[0]
    data_comb   = fits.getdata(comb_img)
    data_comb_f = data_comb+np.mean(im_mean_bkg)
    #Save combined image with added mean bkg
    fits.writeto(comb_img,data_comb_f,hdu_comb.header,overwrite=True)
    #Show combined image in ds9
    showinds9(comb_img)
    
#If images do not have WCS
elif shift_imgs[0]=='0':
    imfiles2=[]
    for i in range(1,len(shift_imgs)):
        imfiles2.append(shift_imgs[i])
    #Save scaled aligned images to list to then combine them
    with open('input_combine.lst','w+') as outfile:
        outfile.write('\n'.join(imfiles2))
        outfile.write('\n')
    #Combine images 
    comb_img = comb(imfiles2,'input_combine.lst')
    #Show combined image in ds9
    showinds9(comb_img)
    
#Move pixel masks to pix_mask folder
print('Pixel masks moved to  pix_mask folder')
if not os.path.exists('pix_mask'):
    os.system("mkdir %s"%'pix_mask')
os.system('mv *'+filt.lower()+'*.pl pix_mask')


#Remove auxiliary files
os.system('rm -f *.coo')
os.system('rm -f *.ctr')
os.system('rm -f *.obj')
os.system('rm -f *_scaled.fits')
os.system('rm -f *_trans0.fits')
os.system('rm -f *_trans.fits')
os.system('rm -f *_trans_scaled.fits')
os.system('rm -f *_trans.fits.ctr')
os.system('rm -f *.fits.ctr')
os.system('rm -f input_combine.lst')
os.system('rm -f fwhm*')
os.system('rm -f geomap.xy')
os.system('rm -f out.geo')
time.sleep(1.)
