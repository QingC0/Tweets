
base= {'sz_w':3, 'hinge_t':0.05, 'log':1, 'attn':1}
expts = ['Eth', 'Age', 'Edu', 'Inc', 'Gen']

conf = dict([(e, base.copy()) for e in expts])
conf['Eth'].update({'n_cnn':40, 'mmC':0.01, 'n_ep':150, 'nacc_T':0.85,'bnorm':1,'lr':2e-4})
conf['Age'].update({'n_cnn':40, 'mmC':0.01, 'n_ep':150, 'nacc_T':0.6, 'bnorm':1,'lr':2e-4})
conf['Edu'].update({'n_cnn':100, 'mmC':0.1, 'n_ep':150, 'nacc_T':0.85, 'bnorm':1,'lr':2e-4})
conf['Inc'].update({'n_cnn':60, 'mmC':0.1, 'n_ep':150, 'nacc_T':0.8, 'bnorm':1, 'lr':2e-4})
conf['Gen'].update({'n_cnn':40, 'mmC':0.01, 'n_ep':150, 'nacc_T':0.9,'bnorm':1, 'lr':2e-4})

