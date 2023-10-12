#!/usr/bin/env python
# coding: utf-8

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/19 13:57
@version: 1.0.0
"""


import re
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml import etree
from title2info import title2info


mags = ['ICLR','AAAI','NIPS','NeurIPS','ACL','ICML','IJCAI','ICLR','ECAI','AISTATS','Nature+Biotechnology',
 'Nature+Chemical+Biology',
 'nature+communications',
 'nature+methods',
 'nature+machine+intelligence',
 'Nature+Reviews+Drug+Discovery',
 'Nature+Computational+Science',
 'science',
 'science+chemical',
 'bioinformatics',
 'brief+in+bioinformatics',
 'Journal+of+Medicinal+Chemistry',
 'Journal+of+Chemical+Information+and+Modeling',
 'ICLR','AAAI','NIPS','NeurIPS','ACL','ICML','IJCAI','ICLR','ECAI','AISTATS']
output_str = ''
xx = 0
page_limit = 10
max_page = 5


path = 'phantomjs.exe'
driver = webdriver.PhantomJS(path)

for m in tqdm(mags):
    driver.get(f'https://scholar.google.com/scholar?as_ylo=2018&q=protein+generation+source:"{m}"&hl=zh-CN&as_sdt=0,5')
    # a = driver.find_element_by_xpath('/html/body/div/div[9]/div[3]/div').text
    content = driver.page_source
    et = etree.HTML(content)
    a = et.xpath('//*[@id="gs_ab_md"]/div/text()')
    if not a:
        print(0, m)
        continue
    a = et.xpath('//*[@id="gs_ab_md"]/div/text()')
    temp = re.findall(' ([\d,]+) ', a[0])
    temp = temp[0]
    temp = int(temp.replace(',', '_'))
    n = temp

    print(n, m)

    for cur_start in range(0, n, 10):
        if cur_start / page_limit >= max_page:
            break
        driver.get(
            f'https://scholar.google.com/scholar?start={cur_start}&as_ylo=2021&q=molecule+generation+source:"{m}"&hl=zh-CN&as_sdt=0,5')
        l = driver.find_element_by_xpath(f'/html/body/div/div[10]/div[2]/div[3]/div[2]').find_elements_by_class_name(
            'gs_ri')
        if cur_start + 10 < n:
            xx += len(l) - page_limit
        for li in l:
            a = [i.text for i in li.find_elements_by_xpath("./*")]
            year = re.findall(r'(20\d\d)', a[1])
            if len(year) > 0:
                year = re.findall(r'(20\d\d)', a[1])[0]
            ci = re.findall(r"(\d+)", a[3])
            if ci:
                ci = ci[0]
            else:
                ci = 0
            title = a[0].replace('[HTML] ', '').replace('[PDF] ', '')
            print(title)
            other = title2info(20, title)
            print(other)
            abstract = ''
            volume = ''
            issue = ''
            doi = ''
            page = ''
            if not other['error']:
                abstract = other['abstract']
                page = other['page']
                volume = other['volume']
                issue = other['issue']
                doi = other['doi']
            output_str += f'''
%0 Journal Article
%T {title}
%J {m}
%D {year}
%X {abstract}
%Z 引用:{ci}
%V {volume}
%N {issue}
%P {page}
%R {doi}
'''


with open('scholar.enw','w',encoding='utf-8') as f:
    f.write(output_str)
