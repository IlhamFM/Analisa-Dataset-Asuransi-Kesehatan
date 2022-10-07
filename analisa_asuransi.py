import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from math import sqrt
#from scipy.stats import ttest_ind
    
def uji_t_dua_means(x1, x2, s1, s2, n1, n2):
    z_kritis = 1.645
    uji_t = (x1 - x2)/ sqrt ((pow(s1,2)/n1) + (pow(s2,2)/n2))
    if uji_t > z_kritis:
        kesimpulan = "Tolak Ho"
    else:
        kesimpulan = "Terima H0"
    return [uji_t, kesimpulan]

def konversi_df(uji_list, list_simpulan):
    dict_tagihan_smoker = {
        "z_kritis" : [1.645] * 10,
        "t_uji" : uji_list,
        "kesimpulan" : list_simpulan
    }
    df_hasil_uji = pd.DataFrame(dict_tagihan_smoker)
    return df_hasil_uji
    
def olah_data():
    df = pd.read_csv('insurance.csv')

    #1 - Analisa Descriptive Statistic
    #rerata umur populasi
    umur = df["age"]
    rerata_umur = umur.mean()
    print(f"\nRerata Umur : {rerata_umur:.2f}")

    #rerata bmi perokok	& non perokok
    bmi = df["bmi"]
    perokok = df[df["smoker"] == "yes"]
    non_perokok = df[df["smoker"] == "no"]
    bmi_perokok = perokok["bmi"]
    bmi_non_perokok = non_perokok["bmi"]
    rerata_bmi_perokok = bmi_perokok.mean()
    rerata_bmi_non_perokok = bmi_non_perokok.mean()
    print(f"\nRerata BMI \t\t: {bmi.mean():.2f}")
    print(f"Rerata BMI perokok \t: {rerata_bmi_perokok:.2f}")
    print(f"tRerata BMI non perokok : {rerata_bmi_non_perokok:.2f}")

    # variansi & rerata tagihan
    tagihan_perokok = perokok["charges"]
    rerata_tagihan_perokok = tagihan_perokok.mean()
    var_tagihan_perokok = tagihan_perokok.var()
    std_tagihan_perokok = tagihan_perokok.std()
    tagihan_non_perokok = non_perokok["charges"]
    rerata_tagihan_non_perokok = tagihan_non_perokok.mean()
    var_tagihan_non_perokok = tagihan_non_perokok.var()
    std_tagihan_non_perokok = tagihan_non_perokok.std()
    print("\n\t","tagihan perokok\t", "tagihan non perokok")
    print("{: <5} {: ^19} {: ^32}".format(*["rerata", f"{rerata_tagihan_perokok:.2f}", f"{rerata_tagihan_non_perokok:.2f}"]))
    print("{: <5} {: ^20} {: ^32}".format(*["var", f"{var_tagihan_perokok:.2f}", f"{var_tagihan_non_perokok:.2f}"]))
    print("{: <5} {: ^20} {: ^32}".format(*["std", f"{std_tagihan_perokok:.2f}", f"{std_tagihan_non_perokok:.2f}"]))

    #rerata umur laki2 dan perempuan
    laki2 = df[df["sex"] == "male"]
    perempuan = df[df["sex"] == "female"]
    umur_laki = laki2["age"]
    umur_perempuan = perempuan["age"]
    rerata_umur_laki = umur_laki.mean()
    rerata_umur_perempuan = umur_perempuan.mean()
    baris_umur = "", f"{rerata_umur_laki:.2f}", f"{rerata_umur_perempuan:.2f}"
    laki2_perokok = laki2[laki2["smoker"] == "yes"]
    umur_laki2_perokok = laki2_perokok["age"]
    laki2_non_perokok = laki2[laki2["smoker"] == "no"]
    umur_laki2_non_perokok = laki2_non_perokok["age"]
    perempuan_perokok = perempuan[perempuan["smoker"] == "yes"]
    umur_perempuan_perokok = perempuan_perokok["age"]
    perempuan_non_perokok = perempuan[perempuan["smoker"] == "no"]
    umur_perempuan_non_perokok = perempuan_non_perokok["age"]
    print("\n{: <12} {: ^21} {: ^30}".format(*["", "rerata umur laki-laki", "rerata umur perempuan"]))
    print("{: <5} {: ^33} {: ^20}".format(*["", f"{rerata_umur_laki:.2f}", f"{rerata_umur_perempuan:.2f}"]))
    print("{: <5} {: ^29} {: ^23}".format(*["perokok", f"{umur_laki2_perokok.mean():.2f}", f"{umur_perempuan_perokok.mean():.2f}"]))
    print("{: <5} {: ^22} {: ^29}".format(*["non perokok", f"{umur_laki2_non_perokok.mean():.2f}", f"{umur_perempuan_non_perokok.mean():.2f}"]))

    #print("\t", f"{rerata_tagihan_perokok:.2f}", "\t\t\t", f"{rerata_tagihan_non_perokok:.2f}")

    #rerata tagihan bmi di atas 25
    perokok_bmi_ats_25 = perokok[perokok["bmi"] >= 25]
    perokok_bmi_bwh_25 = perokok[perokok["bmi"] < 25]
    perokok_bmi_under = perokok_bmi_bwh_25[perokok_bmi_bwh_25["bmi"] < 18.5]
    perokok_bmi_normal = perokok_bmi_bwh_25[perokok_bmi_bwh_25["bmi"] >= 18.5]

    non_perokok_bmi_ats_25 = non_perokok[non_perokok["bmi"] >= 25]
    non_perokok_bmi_bwh_25 = non_perokok[non_perokok["bmi"] < 25]
    non_perokok_bmi_under = non_perokok_bmi_bwh_25[non_perokok_bmi_bwh_25["bmi"] < 18.5]
    non_perokok_bmi_normal = non_perokok_bmi_bwh_25[non_perokok_bmi_bwh_25["bmi"] >= 18.5]

    tagihan_perokok_bmi_ats_25 = perokok_bmi_ats_25["charges"]
    tagihan_non_perokok_bmi_ats_25 = non_perokok_bmi_ats_25["charges"]
    print("\nrerata tagihan perokok BMI > 25\t\t", "rerata tagihan non perokok BMI > 25") 
    print("{: ^32} {: ^52}".format(*[f"{tagihan_perokok_bmi_ats_25.mean():.2f}", f"{tagihan_non_perokok_bmi_ats_25.mean():.2f}"]))
    tagihan_perokok_bmi_normal = perokok_bmi_normal["charges"]
    tagihan_non_perokok_bmi_normal = non_perokok_bmi_normal["charges"]
    print("\nrerata tagihan perokok bmi normal\t", "rerata tagihan non perokok bmi normal") 
    print("{: ^33} {: ^47}".format(*[f"{tagihan_perokok_bmi_normal.mean():.2f}", f"{tagihan_non_perokok_bmi_normal.mean():.2f}"]))

    #2 - Analisa Variabel Kategorik (PMF)
    #Tagihan jenis kelamin tertinggi
    tagihan_laki = laki2["charges"]
    tagihan_perempuan = perempuan["charges"]
    print("\n\t","   tagihan laki2", "   tagihan perempuan")
    print("{: <8} {: >11} {: >16}".format(*["rerata", f"{tagihan_laki.mean():.2f}", f"{tagihan_perempuan.mean():.2f}"]))
    print("{: <8} {: >11} {: >16}".format(*["maksimal", f"{tagihan_laki.max():.2f}", f"{tagihan_perempuan.max():.2f}"]))
    print("{: <8} {: >10} {: >16}".format(*["median", f"{tagihan_laki.median():.2f}", f"{tagihan_perempuan.median():.2f}"]))
    print("{: <8} {: >10} {: >16}".format(*["min", f"{tagihan_laki.min():.2f}", f"{tagihan_perempuan.min():.2f}"]))
    #print("{: <8} {: >7} {: >12}".format(*["peluang", f"{peluang_tagihan_laki:.2f}", f"{peluang_tagihan_perempuan:.2f}"]))

    #Peluang tagihan region
    southwest = df[df["region"] == "southwest"]
    southeast = df[df["region"] == "southeast"]
    northwest = df[df["region"] == "northwest"]
    northeast = df[df["region"] == "northeast"]
    tagihan_southwest = southwest["charges"]
    tagihan_southeast = southeast["charges"]
    tagihan_northwest = northwest["charges"]
    tagihan_northeast = northeast["charges"]
    plt.hist(tagihan_southwest.to_numpy(), bins=100)
    plt.xlabel("southwest")
    plt.show()
    plt.hist(tagihan_southeast.to_numpy(), bins=100)
    plt.xlabel("southeast")
    plt.show()
    plt.hist(tagihan_northwest.to_numpy(), bins=100)
    plt.xlabel("northwest")
    plt.show()
    plt.hist(tagihan_northeast.to_numpy(), bins=100)
    plt.xlabel("northeast")
    plt.show()

    #Proposisi perokok dan non perokok
    total_tagihan_perokok = perokok["charges"].sum()
    total_tagihan_non_perokok = non_perokok["charges"].sum()
    print("\n{: <8} {: <10} {: <10}".format(* ["", "perokok", "non perokok"]))
    print("{: <8} {: <10} {: <10}".format(* ["jumlah", len(perokok), len(non_perokok)]))
    print("{: <8} {: <10} {: <10}".format(* ["proporsi", f"{len(perokok)/len(df):.2f}", f"{len(non_perokok)/len(df):.2f}"]))
    print("\n{: <8} {: <10} {: <10}".format(* ["tagihan", f"{total_tagihan_perokok:.1f}", f"{total_tagihan_non_perokok:.1f}"]))

    #Peluang jika dia perokok
    perempuan_perokok = perokok[perokok ["sex"] == "female"]
    laki2_perokok = perokok[perokok ["sex"] == "male"]
    print("\n{: <14} {: <10} {: <10}".format(* ["", "perempuan", "laki-laki"]))
    print("{: <14} {: ^10} {: ^9}".format(* ["jika perokok", f"{len(perempuan_perokok)/len(perokok):.2f}", f"{len(laki2_perokok)/len(perokok):.2f}"]))

    #3 - Analisa Variabel Kontinu (CDF)
    #Cek distribusi bmi
    bmi = df["bmi"]
    plt.hist(bmi.to_numpy(), bins=100)
    plt.xlabel("bmi")
    #plt.show()

    #Peluang bmi > 25, tagihan > 16700
    df_bmi_atas_25 = df[df["bmi"] >= 25]
    bmi_25_tghn_16_7k = df_bmi_atas_25[df_bmi_atas_25["charges"] > 16700]
    p_bmi_25_tghn_16_7k = len(bmi_25_tghn_16_7k)/len(df_bmi_atas_25)
    print("\nPeluang bmi >= 25 mendapat tagihan > 16.7k : " + f"{p_bmi_25_tghn_16_7k:.2f}")

    #Peluang tagihan > 16.7k jika perokok
    tagihan_perokok_16_7k = perokok[perokok["charges"] > 16700]
    print("\nPeluang tagihan > 16.7k jika perokok : " + f"{len(tagihan_perokok_16_7k)/len(perokok):.2f}")

    #Peluang bmi < 25, tagihan > 16700
    df_bmi_bwh_25 = df[df["bmi"] < 25]
    bmi_bwh_25_tghn_16_7k = df_bmi_bwh_25[df_bmi_bwh_25["charges"] > 16700]
    p_bmi_bwh_25_tghn_16_7k = len(bmi_bwh_25_tghn_16_7k) / len(df_bmi_bwh_25)
    print("\n==========================")
    print("Peluang tagihan > 16.7k ")
    print("==========================")
    print("{: <12} {: ^12} ".format(*["bmi < 25", "bmi >= 25"]))
    print("{: ^9} {: ^18} ".format(*[ f"{p_bmi_bwh_25_tghn_16_7k:.2f}", f"{p_bmi_25_tghn_16_7k:.2f}"]))

    #Peluang perokok & non perokok, tagihan > 16.7k
    perokok_bmi_25 = perokok[perokok["bmi"] > 25]
    non_perokok_bmi_25 = non_perokok[non_perokok["bmi"] > 25]
    perokok_bmi_25_tghn_16_7k = perokok_bmi_25[perokok_bmi_25["charges"] > 16700]
    non_perokok_bmi_25_tghn_16_7k = non_perokok_bmi_25[non_perokok_bmi_25["charges"] > 16700]
    p_perokok_bmi_25_tghn_16_7k = len(perokok_bmi_25_tghn_16_7k) / len(perokok_bmi_25)
    p_non_perokok_bmi_25_tghn_16_7k = len(non_perokok_bmi_25_tghn_16_7k) / len(non_perokok_bmi_25)
    print("\n====================================")
    print("Peluang tagihan > 16.7k & bmi >= 25")
    print("====================================")
    print("{: <15} {: ^15} ".format(*["perokok", "non_perokok"]))
    print("{: ^8} {: ^29} ".format(*[ f"{p_perokok_bmi_25_tghn_16_7k:.2f}", f"{p_non_perokok_bmi_25_tghn_16_7k:.2f}"]))

    #4 - Analisa Korelasi Variabel
    df["sex"] = df["sex"].replace(["female", "male"], [0, 1])
    df["smoker"] = df["smoker"].replace(["yes", "no"], [1, 0])
    df["region"] = df["region"].replace(["southwest", "southeast", "northwest", "northeast"], [0.0, 0.1, 1.0, 1.1])
    print()
    print(df.corr())
    f,ax = plt.subplots(figsize=(9,6))
    sns.heatmap(df.corr(), annot = True, linewidths = 1.5, fmt = ".2f", ax = ax)
    plt.show()

    #5. Uji Hipotesis
    #5. 1. Tagihan perokok > non perokok
    # x1 = perokok 
    # x2 = non perokok
    # H0 mu1 <= mu2
    # Ha mu1 > mu2 
    # tingkat signifikansi = 0,05
    #z_kritis = 1.645
    list_t_smoker = []
    list_kesimpulan_smoker = []
    for i in range (10):
        sample_tghn_perokok = tagihan_perokok.sample(n=25)
        sample_tghn_non_perokok = tagihan_non_perokok.sample(n=25)
        uji_t, kesimpulan = uji_t_dua_means(sample_tghn_perokok.mean(), sample_tghn_non_perokok.mean(), sample_tghn_perokok.std(), 
            sample_tghn_non_perokok.std(), len(sample_tghn_perokok), len(sample_tghn_non_perokok))
        list_t_smoker.append(uji_t)
        list_kesimpulan_smoker.append(kesimpulan)
    print("\nH0 : Tagihan kesehatan perokok <= Tagihan kesehatan non perokok")
    print("Ha : Tagihan kesehatan perokok > Tagihan kesehatan non perokok")
    print(konversi_df(list_t_smoker, list_kesimpulan_smoker))

    # Interpretasi Hasil
    #5. 2. Tagihan BMI>=25 > BMI<25
    # x1 = bmi >= 25 
    # x2 = bmi < 25
    # H0 mu1 <= mu2
    # Ha mu1 > mu2 
    # tingkat signifikansi = 0,05
    #z_kritis = 1.645
    tghn_bmi_atas_25 = df_bmi_atas_25["charges"]
    tghn_bmi_bwh_25 = df_bmi_bwh_25["charges"]
    list_t_bmi = []
    list_kesimpulan_bmi = []
    for i in range (10):
        sample_bmi_atas_25 = tghn_bmi_atas_25.sample(25)
        sample_bmi_bwh_25 = tghn_bmi_bwh_25.sample(25)
        uji_t_bmi, kesimpulan_bmi= uji_t_dua_means(sample_bmi_atas_25.mean(), sample_bmi_bwh_25.mean(), sample_bmi_atas_25.std(), 
            sample_bmi_bwh_25.std(), len(sample_bmi_atas_25), len(sample_bmi_bwh_25))
        list_t_bmi.append(uji_t_bmi)
        list_kesimpulan_bmi.append(kesimpulan_bmi)
    print("\nH0 : Tagihan kesehatan BMI diatas sama dengan 25 <= tagihan BMI dibawah 25")
    print("Ha : Tagihan kesehatan BMI diatas  25 > tagihan BMI dibawah 25")
    print(konversi_df(list_t_bmi, list_kesimpulan_bmi))

    #5. 3. Tagihan laki2> perempuan
    # x1 = bmi >= 25 
    # x2 = bmi < 25
    # H0 mu1 <= mu2
    # Ha mu1 > mu2 
    # tingkat signifikansi = 0,05
    #z_kritis = 1.645
    t_list_sex = []
    list_kesimpulan_sex = []
    for i in range (10):
        sample_tghn_laki = tagihan_laki.sample(25)
        sample_tghn_perempuan = tagihan_perempuan.sample(25)
        uji_t_sex, kesimpulan_sex = uji_t_dua_means(sample_tghn_laki.mean(), sample_tghn_perempuan.mean(), sample_tghn_laki.std(), 
            sample_tghn_perempuan.std(), len(sample_tghn_laki), len(sample_tghn_perempuan))
        t_list_sex.append(uji_t_sex)
        list_kesimpulan_sex.append(kesimpulan_sex)
    print("\nH0 : Tagihan kesehatan laki-laki <= perempuan")
    print("Ha : Tagihan kesehatan laki-laki > perempuan")
    print(konversi_df(t_list_sex, list_kesimpulan_sex))

olah_data()