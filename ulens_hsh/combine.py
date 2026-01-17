#Delete files from possible previous run from the list
def remove(filt):
    os.system('rm -f *'+filt.lower()+'*_trans0.fits')
    os.system('rm -f *'+filt.lower()+'*_trans.fits')
    os.system('rm -f *'+filt.lower()+'*_scaled.fits')
    os.system('rm -f *'+filt.lower()+'_comb.fits')
    os.system('rm -f *'+filt.lower()+'_comb2.fits')
    print()
    return()

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
    '''     
    #Select coordinates in reference image 
    coordfile1=select_xy(ref_img)
    #Center reference image
    out_cen1=center_e(str(mpath.parents[2])+'/scripts/',ref_img,coordfile1)
    '''
    ref_coo_file = str(mpath.parents[1])+'/objetos/'+objname+'/'+objname+'_refpsf.coo'
    if os.path.exists(ref_coo_file):
        print(f"Using existing reference star file: {ref_coo_file}")
        coordfile1 = ref_coo_file
    else:
        print("Reference file not found. Please select stars manually in DS9.")
        coordfile1 = select_xy(ref_img)

    out_cen1 = center_e(str(mpath.parents[2])+'/scripts/', ref_img, coordfile1)

    
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
            coordfile2 = coordfile1 # coordfile2=select_xy(imfiles[0][i])
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
def algn_with_some_wcs(imfiles,crv, objname):
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
    '''
    #Select coordinates in reference image 
    coordfile1=select_xy(ref_img)
    #Center reference image
    out_cen1=center_e(str(mpath.parents[2])+'/scripts/',ref_img,coordfile1)
    '''

    # ================================
    # AUTO-GENERATE .COO IF REQUESTED
    # ================================
    auto = "A" #input("\nPress A for automatic reference-star .coo using Gaia, else ENTER: ")
    
    if auto.upper() == "A":
        print("\nUsando generación automática con Gaia...")
    
    # Calcula radio ~mitad FOV
    hdul = fits.open(imfiles[0][0])
    ny, nx = hdul[0].data.shape
    try:
        cdelt = np.mean(np.abs([hdul[0].header['CDELT1'], hdul[0].header['CDELT2']])) * 60  # arcmin/pix
    except:
        cdelt = 0.53  # fallback HSH arcsec/pix
    hdul.close()
    search_radius_arcmin = (min(nx, ny) * cdelt / 60) / 2
    
    imdir = os.getcwd()
    ra_obj, dec_obj = load_target_coordinates(imdir)
    print(f"Target: RA={ra_obj:.6f} Dec={dec_obj:.6f}")
    
    ref_coo_file = str(mpath.parents[1])+'/objetos/'+objname+'/'+objname+'_refpsf.coo'	
    if not os.path.exists(ref_coo_file):
        print(f"Usando refs automático: {ref_coo_file}")
        try:
            stars_auto = generate_ref_psf_coo(objname, ra_obj, dec_obj, imdir, lista, 
                         fov_frac=0.3, min_mag=9, max_mag=12, plot=True)
            coordfile1 = ref_coo_file
        except:
            print("Generación automática falló. Selecciona manual en DS9.")
            ref_img = wcs_img[0] if 'wcs_img' in locals() else imfiles[0][0]
            coordfile1 = select_xy(ref_img)
    else:
        print("Archivo de objetos de referncia para alineación existente. Eliminar si se quiere recargar")
        coordfile1 = ref_coo_file
    
    out_cen1 = center_e(str(mpath.parents[2])+'/scripts/', ref_img if 'ref_img' in locals() else wcs_img[0], coordfile1)

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
            coordfile2 = coordfile1 #coordfile2=select_xy(imfiles[0][i])
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
    #ETIME = [i.header['EXPTIME'] for i in hdu_list]
    #etrem_indx = []
    #for i in range(len(ETIME)):
    #    if band == 'B':
    #        if ETIME[i] != 90.:
    #            etrem_indx.append(i)
    #            print('Image ',imfiles[i])
    #            print('will be removed due to bad EXPTIME')
    #    else:
    #        if ETIME[i] != 60.:
    #            etrem_indx.append(i)
    #            print('Image ',imfiles[i])
    #            print('will be removed due to bad EXPTIME')
    #Reproject ~ align based on WCS solution
    for i in range(len(imfiles)):
        if i != ref_index: # and i not in etrem_indx:
            hdu2 = hdu_list[i]
            array,footprint = reproject_interp(input_data=hdu2,
                                              output_projection=hdu1.header)
            #Save aligned images into fits files
            fits.writeto(imfiles[i][:-5]+'_trans.fits', array, hdu1.header,
                         overwrite=True)
    #Create and save the combination list with the aligned images
    aligned_imgs = [imfiles[i][:-5]+'_trans.fits'
                    for i in range(len(imfiles)) 
                    if imfiles[i]!=imfiles[ref_index]] # i not in etrem_indx
                   # and imfiles[i]!=imfiles[ref_index]]
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
    #sncatcoord=pd.read_csv(str(mpath.parents[1])+'/objetos/'+obj+'/'+obj\
    #                       +'.coo',header=None,delim_whitespace=True,
    #                       comment='#')
    #snradeg  = str(float(sncatcoord[0]))
    #sndecdeg = str(float(sncatcoord[1]))
    df = pd.read_csv(str(mpath.parents[1])+'/objetos.csv')
    snradeg,sndecdeg = df[df["objeto"]=="OGLE-2025-BLG-0451"][["ra_deg", "dec_deg"]].values[0]
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