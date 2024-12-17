---
title: "SFTP Connect Remote Server"
date: 2024-12-17T22:54:25+08:00
description: SFTP Connect Remote Server
menu:
  sidebar:
    name: SFTP
    identifier: sftp
    parent: tips
    weight: 2003
hero: /images/lake.jpg
tags:
- SFTP
- Remote Server
categories:
- Basic
---

```python
sftp username@remote_url
# input password
cd # to a directory
ls # list the files in that dir
put filename # upload "filename" in local to remote server
# or
put path/your_file # the path of your local file
```