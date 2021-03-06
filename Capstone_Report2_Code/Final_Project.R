library(factoextra)
library(robustHD)



# conduct PCA
data = read.csv("~/Desktop/data.csv")
dim(data)
colname = c('number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass',
                   'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass',
                   'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass',
                   'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie',
                   'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie',
                   'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie',
                   'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius',
                   'wtd_gmean_atomic_radius', 'entropy_atomic_radius',
                   'wtd_entropy_atomic_radius', 'range_atomic_radius',
                   'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius',
                   'mean_Density', 'wtd_mean_Density', 'gmean_Density',
                   'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density',
                   'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density',
                   'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity',
                   'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
                   'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
                   'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
                   'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat',
                   'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat',
                   'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat',
                   'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat',
                   'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity',
                   'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
                   'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
                   'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
                   'std_ThermalConductivity', 'wtd_std_ThermalConductivity',
                   'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence',
                   'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence',
                   'range_Valence', 'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence')
colnames(data) = colname
data_pca = princomp(data, cor = T)
summary(data_pca)
data_pca$loadings

fviz_contrib(data_pca, choice="var", axes = 1, top = 5)

# scree plot
plot(1:10, (data_pca$sdev[1:10])^2, type='b', 
     main="Scree Plot", xlab="Number of Components", ylab="Eigenvalue Size")

# biplot
fviz_pca_var(data_pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE,# Avoid text overlapping
             select.var = list(contrib = 10)
)

