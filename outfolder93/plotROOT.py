from ROOT import TFile
from ROOT import TH1F
from ROOT import TCanvas
from ROOT import TPaveStats
from ROOT import TLegend
import ROOT
import numpy as np


m_predM900 = TH1F("m_pred900", "M Predicted M900", 100, 0, 2)
m_predM900.SetLineColor(1)
m_predM900.SetLineWidth(2)
m_predM900.SetLineStyle(1)
m_predM1200 = TH1F("m_pred1200", "M Predicted M1200", 100, 0, 2)
m_predM1200.SetLineColor(2)
m_predM1200.SetLineWidth(2)
m_predM1200.SetLineStyle(1)
m_predM600 = TH1F("m_pred600", "M Predicted M600", 100, 0, 2)
m_predM600.SetLineColor(3)
m_predM600.SetLineWidth(2)
m_predM600.SetLineStyle(1)
canvas = TCanvas("combined", "combined", 1200, 600)
pred900 = np.load("predvM900.pickle", allow_pickle=True)
pred1200 = np.load("predvM1200.pickle", allow_pickle=True)
pred600 = np.load("predvM600.pickle", allow_pickle=True)
# energies = np.load("totalRechitEnergies.pickle", allow_pickle=True)
# pred = [i - 15 / 16 * mean_pred for i in pred]
# gen = [i - 15 / 16 * mean_gen for i in gen]
for i in range(len(pred900)):
    m_predM900.Fill(pred900[i])
for i in range(len(pred1200)):
    m_predM1200.Fill(pred1200[i])
for i in range(len(pred600)):
    m_predM600.Fill(pred600[i])
legend = TLegend(0.75, 0.25, 0.9, 0.4)
legend.SetHeader("", "C")
legend.AddEntry(m_predM600, "M Predicted M600", "l")
legend.AddEntry(m_predM1200, "M Predicted M1200", "l")
legend.AddEntry(m_predM900, "M Predicted M900", "l")
canvas.cd()

canvas.Modified()
canvas.Update()
#m_predM600.Draw("same")
#m_predM900.Draw("same")
#m_predM1200.Draw("same")

t = ROOT.RooRealVar("t", "t", 0.4, 0.8)
dh = ROOT.RooDataHist("dh", "dh", [t], Import=m_predM600)
m0 = ROOT.RooRealVar("mean_bw", "mean bg", 91, 0, 200)
sg = ROOT.RooRealVar("sigma_bw", "sigma bg", 10.27, 0.1, 20)
bw = ROOT.RooBreitWigner("bw", "bw", t, m0, sg)
bw.fitTo(dh)

# Plot data, pdf, landau (X) gauss pdf
frame = t.frame(Title="BW fit")
dh.plotOn(frame)
# bw.plotOn(frame, LineStyle="--")
# cball.plotOn(frame, LineStyle="--")
bw.plotOn(frame)
bw.paramOn(frame)

#legend.Draw("same")
#m_predM900.Write()
#m_predM1200.Write()
#m_predM600.Write()
"""
legend = ROOT.TPaveLabel(
    0.75,
    0.7 - 0.30,
    0.89,
    0.89 - 0.30,
    "mean_bw=" + str(m0.getValV()) + "\nalpha_cb=" + str(al.getValV()),
    #    + "\n mean="
    #    + str(lxg.mean())
    #    + "\nstd="
    #    + str(lxg.std()),
    "NDC",
)
"""
frame.GetYaxis().SetTitleOffset(1.4)
frame.Draw()

canvas.Write()
canvas.SaveAs("combined.root")
