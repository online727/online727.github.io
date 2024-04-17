---
layout: archive
title: "个人简历"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

<p align="center">
  <a href="../files/cv.pdf">CV</a> | <a href="../files/school report.pdf">成绩单</a>
</p>

教育经历
======
* <div style="display: flex; justify-content: space-between;">
    <span>南京⼤学</span>
    <span>⾦融学</span>
    <span>本科</span>
    <span>2021.09 - 2025.06</span>
    </div>
  * GPA：4.58 / 5.0
  * 相关课程：Python 程序设计、数据结构与算法分析、机器学习、概率论与数理统计、常微分⽅程、随机过程、⾏为⾦融学、证券投资学、⾦融⼯程学、固定收益证券

技能及荣誉
======
* **编程语⾔**：Python、Pytorch、C、MATLAB、R、LaTeX、Git
* **技能证书**：WorldQuant Alphathon Gold Certificate；WorldQuant Consultant；Coursera Machine Learning Certificate
* **语⾔成绩**：IELTS 7；CET - 6 - 612
* **奖学⾦**：国家励志奖学⾦ * 2，恒芳奖学⾦

实习经历
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>

研究经历
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

项目经历
======
  <ul>{% for post in site.portfolio reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>


<!-- Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Currently signed in to 43 different slack teams -->
