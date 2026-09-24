"""Manifest construction for the OCT/OCTA dataset.

Resolves raw DICOM files on the acquisition drive against the lab's Excel scan
logs, producing patients/scans/frames tables that downstream code (training,
annotation, analysis) reads instead of re-deriving identity from filenames.
"""
