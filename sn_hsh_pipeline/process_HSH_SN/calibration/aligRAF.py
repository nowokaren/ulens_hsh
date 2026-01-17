#!/usr/bin/env python
# coding: utf-8

# # Funciones que implementan las tareas de iraf: 1-center, 2-geomap, 3-geotran, 4-imcombine

# In[1]:


#1-
def center_e(path,im,coords):
    import os
    path   = path
    image  = 'image='+im+' '
    coords = 'coords='+coords+' '
    c1     = 'center '
    c2     = 'output='+im+'.ctr '
    c3     = 'interactive=no '
    c4     = 'cache=no '
    c5     = 'verbose=no '
    c6     = 'verify=no '
    c7     = 'update=no '
    c8     = 'radplots=no '
    c9     = 'wcsin=logical '
    c10    = 'wcsout=logical '
    c11    = 'icommands.p_filename= '
    c12    = 'datapars.ccdread= '
    c13    = 'datapars.obstime= '
    c14    = 'datapars.otime= '
    c15    = 'graphics= '
    c16    = 'display= '
    c17    = 'plotfile= '
    c18    = 'datapars.fwhmpsf=2.5 '
    c19    = 'datapars.scale=0.5 '
    c20    = 'datapars.emission=yes '
    c21    = 'datapars.datamin=INDEF '
    c22    = 'datapars.datamax=INDEF '
    c23    = 'datapars.sigma=INDEF '
    c24    = 'datapars.noise=poisson '
    c25    = 'datapars.gain=GAIN '
    c26    = 'datapars.epadu=1. '
    c27    = 'datapars.readnoise=0. '
    c28    = 'datapars.exposure=EXPTIME '
    c29    = 'datapars.itime=1. '
    c30    = 'datapars.airmass=AIRMASS '
    c31    = 'datapars.xairmass=INDEF '
    c32    = 'datapars.filter=FILTER '
    c33    = 'datapars.ifilter=INDEF '
    c34    = 'centerpars.calgorithm=centroid '
    c35    = 'centerpars.cbox=5. '
    c36    = 'centerpars.cthreshold=0. '
    c37    = 'centerpars.minsnratio=1. '
    c38    = 'centerpars.cmaxiter=10 '
    c39    = 'centerpars.maxshift=1. '
    c40    = 'centerpars.clean=no '
    c41    = 'centerpars.rclean=1. '
    c42    = 'centerpars.rclip=2. '
    c43    = 'centerpars.kclean=3. '
    c44    = 'centerpars.mkcenter=no '    
    c      = c1+image+coords+c2+c3+c4+c5+c6+c7+c8+c9+c10+c11+c12+c13+c14+c15+c16+c17+c18+c19+c20+c21+c22+c23+c24+c25+c26+c27+c28+c29+c30+c31+c32+c33+c34+c35+c36+c37+c38+c39+c40+c41+c42+c43+c44
    
    os.system(path+'x_apphot.e'+' '+c)
    print('center')
    print(im+'.ctr')
    return(im+'.ctr')

#2-
def geomap_e(path,xycoords,out):
    import os
    path     = path
    coords   = 'input='+xycoords+' '
    database = 'database='+out+' '
    gm1      = 'geomap '
    gm2      = 'xmin=1. '
    gm3      = 'xmax=1024. '
    gm4      = 'ymin=1. '
    gm5      = 'ymax=1024. '
    gm6      = 'transforms= '
    gm7      = 'results= '
    gm8      = 'fitgeometry=rxyscale '
    gm9      = 'function=polynomial '
    gm10     = 'xxorder=2 '
    gm11     = 'xyorder=2 '
    gm12     = 'xxterms=half '
    gm13     = 'yxorder=2 '
    gm14     = 'yyorder=2 '
    gm15     = 'yxterms=half '
    gm16     = 'maxiter=0 '
    gm17     = 'reject=3. '
    gm18     = 'calctype=real '
    gm19     = 'verbose=no '
    gm20     = 'interactive=no '
    gm21     = 'graphics=stdgraph '
    gm       = gm1+coords+database+gm2+gm3+gm4+gm5+gm6+gm7+gm8+gm9+gm10+gm11+gm12+gm13+gm14+gm15+gm16+gm17+gm18+gm19+gm20+gm21
    
    os.system(path+'x_images.e'+' '+gm)
    print('Geomap')
    print(out)
    return

#3-
def geotran_e(path,in_img_gt,out_img_gt,database,transforms):
    import os
    path       = path
    in_img_gt  = 'input='+in_img_gt+' '
    out_img_gt = 'output='+out_img_gt+' '
    db         = 'database='+database+' '
    tfrm       = 'transforms='+transforms+' '
    gt1        = 'geotran '
    gt2        = 'geometry=geometric '
    gt3        = 'xmin=INDEF '
    gt4        = 'xmax=INDEF '
    gt5        = 'ymin=INDEF '
    gt6        = 'ymax=INDEF '
    gt7        = 'xscale=1. '
    gt8        = 'yscale=1. '
    gt9        = 'ncols=INDEF '
    gt10       = 'nlines=INDEF '
    gt11       = 'xin=INDEF '
    gt12       = 'yin=INDEF '
    gt13       = 'xshift=INDEF '
    gt14       = 'yshift=INDEF '
    gt15       = 'xout=INDEF '
    gt16       = 'yout=INDEF '
    gt17       = 'xmag=INDEF '
    gt18       = 'ymag=INDEF '
    gt19       = 'xrotation=INDEF '
    gt20       = 'yrotation=INDEF '
    gt21       = 'interpolant=linear '
    gt22       = 'boundary=nearest '
    gt23       = 'constant=0. '
    gt24       = 'xsample=1. '
    gt25       = 'ysample=1. '
    gt26       = 'fluxconserve=yes '
    gt27       = 'nxblock=512 '
    gt28       = 'nyblock=512 '
    gt29       = 'verbose=no '
    gt=gt1+in_img_gt+out_img_gt+db+tfrm+gt2+gt3+gt4+gt5+gt6+gt7+gt8+gt9+gt10+gt11+gt12+gt13+gt14+gt15+gt16+gt17+gt18+gt19+gt20+gt21+gt22+gt23+gt24+gt25+gt26+gt27+gt28+gt29
    os.system(path+'x_images.e'+' '+gt)
    print('Geotran')
    print(out_img_gt)
    return


# In[4]:


#4-
def imcombine_e(path,input_list,out_img_ic):
    import os
    path       = path
    in_list    = 'input=@'+input_list+' '
    out_im_ic  = 'output='+out_img_ic+' '
    ic1        = 'imcombine '
    ic2        = 'headers= '
    ic3        = 'bpmasks= '
    ic4        = 'rejmasks= '
    ic5        = 'nrejmasks='+out_img_ic+'_rejected '
    ic6        = 'expmasks='+out_img_ic+'_exposure ' 
    ic7        = 'sigmas= '
    ic8        = 'logfile=STDOUT '
    ic9        = 'project=no '
    ic10       = 'combine=lmedian '
    ic11       = 'reject=sigclip '
    ic12       = 'blank=0. '
    ic13       = 'expname=EXPTIME '
    ic14       = 'statsec= '
    ic15       = 'gain=GAIN '
    ic16       = 'rdnoise=RDNOISE '
    ic17       = 'snoise=0. '
    ic18       = 'lthreshold=INDEF '
    ic19       = 'hthreshold=INDEF '
    ic20       = 'lsigma=3. '
    ic21       = 'hsigma=3. '
    ic22       = 'pclip=-0.5 '
    ic23       = 'nlow=1 '
    ic24       = 'nhigh=1 '
    ic25       = 'nkeep=1 '
    ic26       = 'grow=0. '
    ic27       = 'mclip=yes '
    ic28       = 'sigscale=0. '
    ic29       = 'offsets=none '
    ic30       = 'outlimits= '
    ic31       = 'imcmb=$I '
    ic32       = 'outtype=real '
    ic33       = 'masktype=none '
    ic34       = 'maskvalue=0 '
    ic35       = 'scale=exposure '
    ic36       = 'zero=mode '
    ic37       = 'weight=none '    
    ic=ic1+in_list+out_im_ic+ic2+ic3+ic4+ic5+ic6+ic7+ic8+ic9+ic10+ic11+ic12+ic13+ic14+ic15+ic16+ic17+ic18+ic19+ic20+ic21+ic22+ic23+ic24+ic25+ic26+ic27+ic28+ic29+ic30+ic31+ic32+ic33+ic34+ic35+ic36+ic37
    os.system(path+'x_images.e'+' '+ic)
    print('Imcombine')
    print(out_img_ic)
    return
