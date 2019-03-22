---
title: Vim Remap JK to Esc in Insert Mode
date: 2019-03-22 19:09:20
auto_excerpt:
  enable: true
  length: 150
categories: Vim 
---

<blockquote class="blockquote-center">inoremap jk &lt;Esc&gt; </blockquote>

In vim when we are in insert mode, pressing <ESC> can be too far away from the keyboard and causing inconvenience. Luckly, we can edit .vimrc file to remap <ESC> key to something else. Following code will remap jk to do the job of <ESC> key in insert mode. 
