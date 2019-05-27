#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 01:27:17 2018

Inter-event correlation analysis

@author: liuxingyu
"""

ic_map4each_event = False
event_prototype_relative = False
event_recall = False
further_analysis = False
event_prototype_absolute_and_fc = False



if ic_map4each_event:
    import os
    import numpy as np
    import nibabel as nib
    import fctool
    from scipy import stats
    import matplotlib.pyplot as plt
    
#    project = 'studyforrest'
    project = 'sherlock'
#    project = 'twilight'
    
    norepeated = '_norepeated' 
    zscore_in_event = 0
    
    roi = 1
    v1_regressed = 0
    
    if project=='studyforrest':
        tr_shift = 2
    else:
        tr_shift = 3

    analysis_dir_parent = ('/nfs/s2/userhome/liuxingyu/workingdir/'
                           'event_cognition')
    ic_ts_path = ('/nfs/e5/{0}/subavg/audiovisual3T/concatenated/'
                  'f.nii.gz'.format(project))
    event_path = os.path.join(analysis_dir_parent, project, 'documents', 
                              'event_boundaries/concat{0}_TR.npy'.format(
                                      norepeated))
    roi_path = ('/nfs/s2/userhome/liuxingyu/workingdir/atlas/MMP1.0_mni152/'
                'HCP-MMP1_on_MNI152_ICBM2009a_nlin_3mm.nii.gz')

    #-----------load data-----------------
    ic_ts = nib.load(ic_ts_path).get_data()
    
    if roi:
        roi_index = nib.load(roi_path).get_data()      
        roi_ts = fctool.roiing_volume_roi_mean(roi_index,ic_ts)
       
        if v1_regressed == False:      
            ic_ts = roi_ts
        else:
            v1 = roi_ts[0,0,0,:].reshape(-1,1)
            roi_ts_v1regressed = np.zeros(np.shape(roi_ts))
            for i in range(np.shape(roi_ts)[0]):
                observed_y = roi_ts[i,0,0,:].reshape(-1,1)
                roi_ts_v1regressed[i,:,:,:] = fctool.residual(
                        v1, observed_y)[:,0]
            ic_ts = roi_ts_v1regressed
            
        
    event_boundary = np.load(event_path)
     
    event_length = np.zeros(np.shape(event_boundary))
    event_length[1:] = event_boundary[1:] - event_boundary[:-1]
    event_length[0] = event_boundary[0]
    
    if zscore_in_event:
        for i in np.arange(1, np.size(event_boundary)-1, 1):  
            # excluding the first and the last event
            event = ic_ts[:,:,:,(event_boundary[i]+tr_shift):
                (event_boundary[i+1]+tr_shift)] 
            event = stats.zscore(event, axis=-1)
            ic_ts[:,:,:,(event_boundary[i]+tr_shift):
                (event_boundary[i+1]+tr_shift)] = event
                
#============== relative event========================    
    event_length_min = 20
    event_length_max = 900000
    fraction_number = 20

    #----------average relative fraction of event--------------
    event_prototype = []
    for i in np.arange(1, np.size(event_boundary)-1, 1):  
        event_x_prototype = []
        # excluding the first and the last event
        if (event_boundary[i+1] - event_boundary[i] >= event_length_min) and (
                event_boundary[i+1] - event_boundary[i] < event_length_max):
            for n in np.arange(0,fraction_number,1):
                unit = (event_boundary[i+1] - event_boundary[i]) / (
                        fraction_number)
                fraction_start = np.int(np.around(
                        event_boundary[i]+tr_shift + n*unit,decimals=0))
                fraction_end = np.int(np.around(
                        event_boundary[i]+tr_shift + (n+1)*unit, decimals=0))
                event = ic_ts[:,:,:,fraction_start:fraction_end]   
                event_x_prototype.append(np.mean(event,axis=-1))
               
            event_prototype.append(event_x_prototype)           
   
    event_prototype = np.asarray(event_prototype,dtype=np.float64)
    event_prototype = np.swapaxes(event_prototype, 1, 2)
    event_prototype = np.swapaxes(event_prototype, 2, 3)
    event_prototype = np.swapaxes(event_prototype, 3, 4)

#    event_prototype_avg = np.mean(event_prototype, axis=0)

 
    
if event_recall:
    from scipy import stats
    
    project = 'sherlock_recall'
    fraction_number = 20
    roi = 1
    ic_ts = nib.load('/nfs/e5/sherlock/subavg/recall/001/'
                    '50eventx{0}fraction_convolved_v2.nii.gz'.format(
                            fraction_number)).get_data()
    roi_path = ('/nfs/s2/userhome/liuxingyu/workingdir/atlas/MMP1.0_mni152/'
                'HCP-MMP1_on_MNI152_ICBM2009a_nlin_3mm.nii.gz')
    
    if roi:
        roi_index = nib.load(roi_path).get_data()      
        not_roi = np.where(roi_index == 0)
        ic_ts[not_roi[0],not_roi[1],not_roi[2],:] = 0
        roi_ts = np.zeros([180,1,1,np.shape(ic_ts)[-1]])
        
        for i in range(1,181,1):
            roi_i_loc = np.where(roi_index==i)
            roi_i = ic_ts[roi_i_loc[0],roi_i_loc[1],roi_i_loc[2],:]
            roi_ts[i-1,0,0,:]= roi_i.mean(0)
        ic_ts = roi_ts
    
    #-------------event prtototype--------
    event_prototype = [ic_ts
                       [:,:,:,fraction_number*i:fraction_number*(i+1)] for i 
                       in range(50)]
    event_prototype = np.asarray(event_prototype, dtype=np.float64)
    
    mask = np.load('/nfs/s2/userhome/liuxingyu/workingdir/event_cognition/'
                   'sherlock/documents/event_sub_remembered.npy')
    mask[mask<5] = 0
    mask[mask!=0] = 1
    
    event_prototype = np.delete(event_prototype, np.where(mask==0), axis=0)
#    event_prototype = stats.zscore(event_prototype, axis=-1)
    event_prototype_avg = event_prototype.mean(0)
    

            
        
if further_analysis:
    from scipy import stats
    

    # ===========leave_one_out iec=============
    cmp_v1_each = 0
    cmp_v1_mean = 0
    
    x = event_prototype[:,:,0,0,:]
    y = event_prototype[:,:,0,0,:].mean(0) * np.shape(x)[0]
    h1 = [fctool.isc(x[i,:,:], (y - x[i,:,:])/
                     (np.shape(x)[0]-1)) for i in range(np.shape(x)[0])]
    h1 = np.asarray(h1,dtype=np.float64)
    if cmp_v1_each:
        h1_v1 = [fctool.isc(x[i,0,:], (y[0,:] - x[i,0,:])/
                            (np.shape(x)[0]-1)) for i in range(np.shape(x)[0])]
        h1_v1 = np.asarray(h1_v1,dtype=np.float64)
        h1 = [h1[:,i]-h1_v1 for i in range(180)]
        h1 = np.asarray(h1,dtype=np.float64)
        h1 = np.swapaxes(h1, 0, 1)
    h1_mse = np.std(h1, axis=0)/np.sqrt(np.shape(h1)[0])
  
    #------------one sample ttest---------------
    if cmp_v1_mean:
        iec_v1 = h1[:,0].mean()
        iec_ttest = stats.ttest_1samp(h1,iec_v1)
        h1_avg = h1.mean(0) - iec_v1
    else:
        iec_ttest = stats.ttest_1samp(h1,0)
        h1_avg = h1.mean(0)
        
    p_value = np.nan_to_num(iec_ttest[1])
    t_value = np.nan_to_num(iec_ttest[0])
    
    #================multiple test correction===================   
    #-------------hdr correction---------
    hdr_Q = fctool.fdr_correction(p_value)
    t_value[hdr_Q > 0.05] = 0  
    #------------Bfrn correction------
    t_value[p_value > 0.05/(61*73*61)] = 0
    
    #=========community detection==========  
    x = fctool.isfc(ic_ts[:,0,0,:],ic_ts[:,0,0,:])
    resorted = fctool.dendo_community(x)[0]
    resort_index = fctool.dendo_community(x)[1]


    #-------------whole brain pattern clustring ----------------
    #----- don't zscore ts----------------------
    event_concat = np.zeros([180,31*fraction_number])
    for i in range(31):
        event_concat[:,i*fraction_number:
            (i+1)*fraction_number] = event_prototype[i,:,0,0,:]
            
    event_concat_cls = fctool.cluster(event_concat.T,2,2)
    event_prototype_cls = [event_concat_cls[
            fraction_number*i:fraction_number*(i+1)] for i in range(31)]
    
    #---------------event clustering----------------
    z = event_prototype_sm.mean(1)
    event_cls = fctool.cluster(z,31,3)
    
    a = event_prototype_sm[np.where(event_cls==1)[0],:,:].mean(1)
    b = a.mean(0)
    c = a.std(0)/np.sqrt(np.shape(a)[0])
    

#==============save t p value==============================
data_info = nib.load('/nfs/e5/sherlock/sub003/audiovisual3T/001/f.nii')

img = nib.Nifti1Image(t_value, None, data_info.get_header())
nib.save(img, '/nfs/s2/userhome/liuxingyu/workingdir/event_cognition/'
         'combined/{0}_iec_t.nii.gz'.format(project))        


img = nib.Nifti1Image(p_value, None, data_info.get_header())
nib.save(img, '/nfs/s2/userhome/liuxingyu/workingdir/event_cognition/'
         'combined/{0}_iec_p.nii.gz'.format(project))


#===============save t p value of roi======================
t_volume = np.zeros(np.shape(roi_index))

for i in range(1,181,1):
    roi_i_loc = np.where(roi_index==i)
    t_volume[roi_i_loc[0],roi_i_loc[1],roi_i_loc[2]] = t_value[i-1,0,0]
