#!/usr/bin/env python
"""
Thomas Klijnsma
"""

########################################
# Imports
########################################

import os, sys
from os.path import *

import numpy, itertools

from math import exp, log, sqrt, erf
import scipy.signal
from scipy.interpolate import interp1d

import Functions


########################################
# For plotting
########################################

import ROOT
from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kError;")
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas( 'c', 'c', 1000, 800 )

CLeftMargin   = 0.15
CRightMargin  = 0.03
CBottomMargin = 0.15
CTopMargin    = 0.03
def SetCMargins(
    LeftMargin   = CLeftMargin,
    RightMargin  = CRightMargin,
    BottomMargin = CBottomMargin,
    TopMargin    = CTopMargin,
    ):
    c.SetLeftMargin( LeftMargin )
    c.SetRightMargin( RightMargin )
    c.SetBottomMargin( BottomMargin )
    c.SetTopMargin( TopMargin )

def GetPlotBase(
    xMin = 0, xMax = 1,
    yMin = 0, yMax = 1,
    xTitle = 'x', yTitle = 'y',
    SetTitleSizes = True,
    ):
    base = ROOT.TH1F()
    ROOT.SetOwnership( base, False )
    base.SetName( 'genericbase' )
    base.GetXaxis().SetLimits( xMin, xMax )
    base.SetMinimum( yMin )
    base.SetMaximum( yMax )
    base.SetMarkerColor(0)
    base.GetXaxis().SetTitle( xTitle )
    base.GetYaxis().SetTitle( yTitle )
    if SetTitleSizes:
        base.GetXaxis().SetTitleSize( 0.06 )
        base.GetYaxis().SetTitleSize( 0.06 )
    return base

CPlotDir = 'plots'
def SaveC( outname, asPDF=True, asPNG=False, asROOT=False ):
    if not isdir(CPlotDir): os.makedirs( CPlotDir )
    outname = join( CPlotDir, outname.replace('.pdf','').replace('.png','').replace('.root','') )
    if asPDF: c.SaveAs( outname + '.pdf' )
    if asPNG: c.SaveAs( outname + '.png' )
    if asROOT: c.SaveAs( outname + '.proot' )

def TGraphFromArrays( name, xs, ys ):
    Tg = ROOT.TGraph(
        len(xs),
        array( 'd', xs ),
        array( 'd', ys ),
        )
    ROOT.SetOwnership( Tg, False )
    Tg.SetLineWidth(2)
    Tg.yMin = min(ys)
    Tg.yMax = max(ys)
    Tg.SetName(name)
    Tg.name = name
    return Tg


########################################
# Main
########################################

def main():

    for MODESAME in [ True, False ]:

        # ======================================
        # Input

        sigma_pred = 172.7
        scaleError = 9.2
        pdfError   = 7.2

        scaleFn = Functions.Gauss( sigma_pred, scaleError )
        pdfFn = Functions.Gauss( sigma_pred, pdfError )

        # ======================================
        # Create the y-values arrays

        nPoints = 1000

        if MODESAME:
            xAxis = Axis( 0., 2.0*sigma_pred, nPoints )
        else:
            xAxis = Axis( 0., 2.5*sigma_pred, nPoints )
        binWidth = xAxis[1]-xAxis[0]

        yScale = [ scaleFn(x) for x in xAxis ]
        yPdf   = [ pdfFn(x)   for x in xAxis ]


        # ======================================
        # Manual convolution

        if MODESAME:
            convolution_manual = manualConvolution_modeSame( yScale, yPdf )
        else:
            convolution_manual = manualConvolution( yScale, yPdf )

        norm_convolution_manual = sum(convolution_manual) * binWidth
        convolution_manual = [ y / norm_convolution_manual for y in convolution_manual ]


        # ======================================
        # Scipy convolution

        convolution_scipy = scipy.signal.convolve(
            yScale, yPdf,
            mode= 'same' if MODESAME else 'full'
            )

        norm_convolution_scipy = sum(convolution_scipy) * binWidth
        convolution_scipy = [ y / norm_convolution_scipy for y in convolution_scipy ]


        # ======================================
        # Plot

        Tg_scale              = TGraphFromArrays( 'scale',              xAxis, yScale )
        Tg_pdf                = TGraphFromArrays( 'pdf',                xAxis, yPdf )
        Tg_convolution_scipy  = TGraphFromArrays( 'convolution_scipy',  xAxis, convolution_scipy )
        Tg_convolution_manual = TGraphFromArrays( 'convolution_manual', xAxis, convolution_manual )
        Tg_convolution_manual.SetLineStyle(2)


        Tgs = [
            Tg_scale,
            Tg_pdf,
            Tg_convolution_scipy,
            Tg_convolution_manual,
            ]

        colors = itertools.cycle( range( 2, 5 ) + range( 6, 9 ) )
        for Tg in Tgs:
            Tg.color = next(colors)
            Tg.SetLineColor( Tg.color )

        yMinAbs = min([ Tg.yMin for Tg in Tgs ])
        yMaxAbs = max([ Tg.yMax for Tg in Tgs ])
        yMin = yMinAbs - 0.05*(yMaxAbs-yMinAbs)
        yMax = yMaxAbs + 0.05*(yMaxAbs-yMinAbs)


        c.Clear()
        SetCMargins()

        base = GetPlotBase(
            xMin = min(xAxis), xMax = max(xAxis),
            yMin = yMin, yMax = yMax
            )
        base.Draw('P')


        leg = ROOT.TLegend(
            1-c.GetRightMargin()-0.25, 1-c.GetTopMargin()-0.25,
            1-c.GetRightMargin(), 1-c.GetTopMargin(),
            )
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)

        for Tg in Tgs:
            Tg.Draw('LSAME')
            leg.AddEntry( Tg.GetName(), Tg.GetName(), 'l' )

        leg.Draw()

        SaveC( 'convtest_{0}_{1}'.format( 'same' if MODESAME else 'regular', nPoints ) )





def Axis( leftBound, rightBound, nPoints ):
    dx = (rightBound-leftBound) / (nPoints-1)
    return [ leftBound + i*dx for i in xrange(nPoints) ]





def manualConvolution( ys1, ys2 ):

    N = len(ys1)

    ret = []
    for n in xrange(N):
       
        val = 0.
        for m in xrange(N):
            if ( n-m >= 0 and n-m < N ) and ( m >= 0 and m < N ):
                val += ys1[m] * ys2[n-m]

        ret.append(val)

    return ret



def manualConvolution_modeSame( ys1, ys2 ):

    N = len(ys1)

    if N % 2 == 1:
        mid = int(N/2)
    else:
        mid = int((N-1)/2)

    rangeArray = range( -mid, mid )

    ret = []


    for n in rangeArray:
       
        val = 0.
        for m in rangeArray:
            val += CallCentredArrayAt( ys1, m ) * CallCentredArrayAt( ys2, n-m )

        ret.append(val)

    return ret


def CallCentredArrayAt( array, i ):
    N = len(array)

    if N % 2 == 1:
        mid = int(N/2)
    else:
        mid = int((N-1)/2)

    if i > mid - 1:
        return 0
    elif i < -mid:
        return 0
    else:
        return array[i+mid]







########################################
# End of Main
########################################
if __name__ == "__main__":
    main()