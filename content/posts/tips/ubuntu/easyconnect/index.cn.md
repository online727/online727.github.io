---
title: "在 Ubuntu 安装 Easyconnect"
date: 2024-12-03T00:15:25+08:00
description: 在 Ubuntu 安装 Easyconnect
menu:
  sidebar:
    name: Easyconnect
    identifier: easyconnect
    parent: ubuntu
    weight: 2002
hero: /images/lake.jpg
tags:
- Ubuntu
- Softwares
- Easyconnect
categories:
- Basic
---


## 0.1 Download `.deb` File
Downloading address: [https://software.openkylin.top/openkylin/yangtze/pool/all/](https://software.openkylin.top/openkylin/yangtze/pool/all/), search for `easyconnect` on that page and download `easyconnect_7.6.7.3.0_amd64.deb`.

## 0.2 Install
``` bash
sudo dpkg --install easyconnect_7.6.7.3.0_amd64.deb
```

## 0.3 Problems
### 0.3.1 Error Info
We can sign in successfully the first time we installed it. But the connection will be failure once we quitted or restarted our computer.

Error info: the verison is not match with the server, please upgrade...

### 0.3.2 Solving Methods
#### 0.3.2.1 Method 1
Delete `/usr/share/sangfor/EasyConnect/resources/conf/pkg_version.xml`

#### 0.3.2.2 Method 2
Revise the `/usr/share/sangfor/EasyConnect/resources/conf/Version.xml`. The picture below is the revised file: 

{{< img src="/posts/tips/ubuntu/easyconnect/version.png" align="center" title="Revised Version.xml" >}}

{{< vs 3 >}}

## 0.4 Uninstall
``` bash
sudo dpkg --remove  easyconnect
```