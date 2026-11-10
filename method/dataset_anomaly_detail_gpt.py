from torch import nn



class DatasetGPT(nn.Module):
    def __init__(self):
        super(DatasetGPT, self).__init__()
        self.mvtec_anomaly_detail_gpt = {
            "carpet": "discoloration in a specific area,irregular patch or section with a different texture,frayed edges or unraveling fibers,burn mark or scorching",
            "grid": "crooked,cracks,excessive gaps,discoloration,deformation,missing,inconsistent spacing between grid elements,corrosion,visible signs,chipping",
            "leather": "scratches,discoloration,creases,uneven texture,tears,brittleness,damage,seams,heat damage,mold",
            "tile": "chipped,irregularities,discoloration,efflorescence,warping,missing,depressions,lippage,fungus,damage",
            "wood": "knots,warping,cracks along the grain,mold growth on the surface,staining from water damage,wood rot,woodworm holes,rough patches,protruding knots",
            "bottle": "cracked large,cracked small,dented large,dented small,leaking,discolored,deformed,missing cap,excessive condensation,unusual odor",
            "cable": "twisted,knotted cable strands,detached connectors,excessive stretching,dents,corrosion,scorching along the cable,exposed conductive material",
            "capsule": "irregular shape,discoloration coloring,crinkled,uneven seam,condensation inside the capsule,foreign particles,unusually soft or hard",
            "hazelnut": "fungal growth,unusual discoloration,rotten or foul odor emanating,insect infestation,wetness,misshapen shell,unusually thin,contaminants,unusual texture",
            "metal_nut": "cracks,irregular threading,corrosion,missing,distortion,signs of discoloration,excessive wear on contact surfaces,inconsistent texture",
            "pill": "irregular shape,crumbling texture,excessive powder,Uneven coating,presence of air bubbles,disintegration,abnormal specks",
            "screw": "rust on the surface,bent,damaged threads,stripped threads,deformed top,coating damage,uneven grooves,inconsistent size",
            "toothbrush": "loose bristles,uneven bristle distribution,excessive shedding of bristles,staining on the bristles,abrasive texture,irregularities in the shape",
            "transistor": "burn marks,detached leads,signs of corrosion,irregularities in the shape,presence of cracks or fractures,signs of physical trauma,irregularities in the surface texture",
            "zipper": "bent,frayed,misaligned,excessive stiffness,corroded,detaches,loose,warped",
        }

        self.visa_anomaly_detail_gpt = {
            "candle": "cracks or fissures in the wax,Wax pooling unevenly around the wick,tunneling,incomplete wax melt pool,irregular or flickering flame,other,extra wax in candle,wax melded out of the candle",
            "capsules": "uneven capsule size,capsule shell appears brittle,excessively soft,dents,condensation,irregular seams or joints,specks",
            "cashew": "uneven coloring,fungal growth,presence of foreign objects,unusual texture,empty shells,signs of moisture,stuck together",
            "chewinggum": "consistency,presence of foreign objects,uneven coloring,excessive hardness,similar colour spot",
            "fryum": "irregular shape,unusual odor,uneven coloring,unusual texture,small scratches,different colour spot,fryum stuck together,other",
            "macaroni1": "uneven shape ,small scratches,small cracks,uneven coloring,signs of insect infestation,uneven texture,Unusual consistency",
            "macaroni2": "irregular shape,small scratches,presence of foreign particles,excessive moisture,Signs of infestation,small cracks,unusual texture",
            "pcb1": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
            "pcb2": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
            "pcb3": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
            "pcb4": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
            "pipe_fryum": "uneven shape,presence of foreign objects,different colour spot,unusual odor,empty interior,unusual texture,similar colour spot,stuck together",
        }

        self.clinicdb_anomaly_detail_gpt = {
            "ClinicDB": "rounded protrusions,well-circumscribed reddish structure,elevation of irregular margin,extending abnormal tissue mass",
        }

        self.colondb_anomaly_detail_gpt = {
            "ColonDB": "nodular bump,surface bulging,mucosal buildup,localized contour change",
        }

        self.brainmri_anomaly_detail_gpt = {
            "brain_mri": "ring-shaped lesion,bright solid mass,dark cyst,blurry halo,dense round spot",
        }

        self.btad_anomaly_detail_gpt = {
            "01": "diffuse bright region,irregular edge,uneven surface,discoloration,localized texture change",
            "02": "localized irregular region,disruptive elongated area,well-circumscribed structure,irregular narrow band,linear anomaly,stretched imprint,elongated trace",
            "03": "elongated edge-aligned rupture,localized fine line noise,scattered spot-like interference,strip artifact,blocky disruption",
        }

        self.mpdd_anomaly_detail_gpt = {
            "tubes": "angular or rectangular cut,curved or slanted break,irregular rupture,linear tear,foreign-colored tube,deformed patch,deformation",
            "metal_plate": "rust corrosion,surface abrasion,linear scratches,coating loss,irregular patches,discoloration,texture disruption,extensive rusting",
            "connector": "black segment missing,incomplete rear section,missing rear cap,rear unit absent,black component detached,silver part truncated,rear side incomplete,partial loss",
            "bracket_white": "small dents or pits,local dark patches,irregular surface scratches,fine linear abrasions,streaked trace,spots of uneven paint",
            "bracket_brown": "bent metal rod near hook,missing or detached white fastener above bracket",
            "bracket_black": "deformation near attachment holes,scratches,dents near holes,cracks,paint peeling,discoloration",
        }

        self.sdd_anomaly_detail_gpt = {
            "SDD": "elongated cracks,thin irregular fractures,curved breaks,horizontal fracture,edge disruptions",
        }

        self.dagm_anomaly_detail_gpt = {
            "Class1": "local texture blurring, texture disruption by a round dark are, circular circulardark inclusion",
            "Class2": "black curved, short, arc lines, inconsistent with background stripes",
            "Class3": "white or light colored abnormal spots, irregular patches differing from surrounding texture",
            "Class4": "extra irregular lines, extra lines disrupting radial structure, curved thin lines",
            "Class5": "small striped patch, tiny parallel-lined region, localized striped anomaly",
            "Class6": "dark linear streak, elongated dark defect",
            "Class7": "white irregular patch, white damaged area, texture-inconsistent white region",
            "Class8": "extra slender curved lines, irregular thin lines disrupting texture, unexpected thin arcs",
            "Class9": "black circular spot, isolated small black dot, round dark speck",
            "Class10": "extra interfering lines, straight interfering lines, curved interfering lines, disrupted regular arc texture, broken uniform texture pattern",
        }

        self.dtd_anomaly_detail_gpt = {
            "Blotchy_099": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Fibrous_183": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Marbled_078": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Matted_069": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Mesh_114": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Perforated_037": "irregularly shaped foreign objects, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Stratified_154": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Woven_001": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Woven_068": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Woven_104": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Woven_125": "irregularly shaped foreign objects, abnormal color patches, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
            "Woven_127": "irregularly shaped foreign objects, curved or irregular lines, small spot - like anomalies, elongated atypical structures, unusual structural formation",
        }

        self.br35h_anomaly_detail_gpt = {
            "br35h": "irregular density areas, abnormal calcifications, distorted structures, abnormal masses, bone erosions or thickenings, contrast enhancements, skull fractures, tissue hyper or hypodensities",
        }

        self.headct_anomaly_detail_gpt = {
            "headct": "focal density changes, abnormal contour deformity, widened ventricles",
        }

        self.isic_anomaly_detail_gpt = {
            "isic": "irregular pigmented patch, uneven color lesion, abnormal shaped mole, ill-defined pigmented area",
        }

        self.tn3k_anomaly_detail_gpt = {
            "tn3k": "irregular hypoechoic area, well-defined abnormal mass, heterogeneous echogenic lesion, distinct abnormal tissue region, atypical ultrasonic structure",
        }

        '''for cls_name in self.mvtec_anomaly_detail_gpt.keys():
            self.mvtec_anomaly_detail_gpt[cls_name] = (
                self.mvtec_anomaly_detail_gpt[cls_name].split(",")
            )

        for cls_name in self.visa_anomaly_detail_gpt.keys():
            self.visa_anomaly_detail_gpt[cls_name] = (
                self.visa_anomaly_detail_gpt[cls_name].split(",")
            )'''

        self.status_abnormal = {}

        self.split_datasets()


        '''for cls_name in self.mvtec_anomaly_detail_gpt.keys():
            self.status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in
                                         self.mvtec_anomaly_detail_gpt[cls_name]]

        for cls_name in self.visa_anomaly_detail_gpt.keys():
            self.status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in
                                         self.visa_anomaly_detail_gpt[cls_name]]'''

    def split_datasets(self):
        dataset_attrs = [
            'mvtec_anomaly_detail_gpt',
            'visa_anomaly_detail_gpt',
            'clinicdb_anomaly_detail_gpt',
            'colondb_anomaly_detail_gpt',
            'brainmri_anomaly_detail_gpt',
            'btad_anomaly_detail_gpt',
            'mpdd_anomaly_detail_gpt',
            'sdd_anomaly_detail_gpt',
            'dagm_anomaly_detail_gpt',
            'dtd_anomaly_detail_gpt',
            'br35h_anomaly_detail_gpt',
            'headct_anomaly_detail_gpt',
            'isic_anomaly_detail_gpt',
            'tn3k_anomaly_detail_gpt'
        ]
        for attr in dataset_attrs:
            dataset_dict = getattr(self, attr, {})
            if isinstance(dataset_dict, dict):
                for cls_name in dataset_dict:
                    if isinstance(dataset_dict[cls_name], str):
                        dataset_dict[cls_name] = dataset_dict[cls_name].split(",")
                        # Used to train
                        self.status_abnormal[cls_name] = ['{} ' + 'with {}'.format(x) for x in
                                                          dataset_dict[cls_name]]

    def choose_dataset_anomaly_detail_gpt(self, dataset_name, cls_name_todino):
        all_dataset=['MVTecDataset','VisaDataset','ClinicDBDataset','ColonDBDataset','BrainMRIDataset','BTADDataset','MPDDDataset','SDDDataset','DAGMDataset','DTDDataset','Br35hDataset','HEADCTDataset','ISICDataset','TN3KDataset']
        dataset_attrs = [
            'mvtec_anomaly_detail_gpt',
            'visa_anomaly_detail_gpt',
            'clinicdb_anomaly_detail_gpt',
            'colondb_anomaly_detail_gpt',
            'brainmri_anomaly_detail_gpt',
            'btad_anomaly_detail_gpt',
            'mpdd_anomaly_detail_gpt',
            'sdd_anomaly_detail_gpt',
            'dagm_anomaly_detail_gpt',
            'dtd_anomaly_detail_gpt',
            'br35h_anomaly_detail_gpt',
            'headct_anomaly_detail_gpt',
            'isic_anomaly_detail_gpt',
            'tn3k_anomaly_detail_gpt'
        ]
        choose_id = all_dataset.index(dataset_name)
        choose_attr = dataset_attrs[choose_id]
        choose_dict = getattr(self, choose_attr, {})
        if isinstance(choose_dict, dict):
            choose_anomaly_detail_gpt = choose_dict[cls_name_todino]
        return choose_anomaly_detail_gpt

    def forward(self, cls_name_totrain):
        return self.status_abnormal[cls_name_totrain]


