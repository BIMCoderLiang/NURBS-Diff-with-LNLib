/*
 * Author:
 * Copyright (c) 2025/02/04- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 * 
 */

#pragma once

#define DLL_EXPORT __declspec(dllexport)
#define DLL_IMPORT __declspec(dllimport)

#if defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(__CYGWIN__)
#ifdef ND_LNLib_HOME
#define ND_LNLib_EXPORT DLL_EXPORT
#else
#define ND_LNLib_EXPORT DLL_IMPORT
#endif
#else
#define ND_LNLib_EXPORT
#endif

