# Questions for Gavrielle

*Context: we're analysing the 2021 OCTAAgeing data (the data behind Untracht et
al., Sci Rep 2023, 13:1122) from the RGB DICOM exports on the external drive.
We take flow as red − green, find the DEJ with a U-Net, and take a MIP of a
papillary slab from 5 px above to 15 px below the DEJ. As a sanity check, our flow
40–90 px below the DEJ matches your per-scan hand vessel area density in
`2021-05-OCTAAgeing-hand.xlsx` closely (Spearman ρ = 0.94 over 224 scans, 19
participants).*

## The flow data

1. **Which data did your MATLAB MIP script read for the 2023 paper?** The RGB
   DICOMs (structural grey with the flow as a red overlay), or flow-only volumes
   exported from VivoTools? If flow-only, do those files still exist, and could
   we have them?
2. **Could we have the surface-detection and MIP script?** We'd like to rebuild
   your MIPs from our files, and compare your fixed 175 µm cut with our
   U-Net-detected DEJ.
3. **How does VivoTools draw the red overlay?** Is red brightness linear in the
   flow (decorrelation) signal, or is there a colour map, gain or threshold? In
   our files flow runs smoothly from 1 to 255 with no cutoff, and is limited to
   255 minus the grey level.

## The study and the spreadsheets

4. **Is there a foot version of `2021-05-OCTAAgeing-hand.xlsx`?**
5. **Which 11 controls were in the paper's age-matched group (Table S7)?** Our
   control list includes CA, JA and TL, who were only scanned in October 2020
   and aren't in the hand spreadsheet.
6. **Why was DP-11-DC left out of the T2DM group?** The spreadsheet has 12
   diabetic participants; the paper has 11.
7. **When during the 5-minute occlusion was the ischemia scan taken?** It's
   scan 6 in the spreadsheet but isn't described in the paper. In our data, foot
   flow rarely drops during that scan, even in controls.
8. **Were any scans excluded for poor quality,** and is that recorded anywhere?
   The "Quality" columns in the spreadsheet are empty.
9. **Were blood pressure, BMI and HbA1c recorded per participant** for the May
   2021 session, and could we have them? We'd like to include them as covariates,
   and to check whether cuff pressure relative to blood pressure could explain
   the weaker occlusion we see in T2DM hands.
