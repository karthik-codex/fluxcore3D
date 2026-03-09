"""
STEP to STL Converter

This module provides functions to convert STEP files to STL format.
Requires one of the following libraries:
    - cadquery (recommended): pip install cadquery
    - pythonocc-core: conda install -c conda-forge pythonocc-core

Usage:
    from step_to_stl import step_to_stl
    step_to_stl("input.step", "output.stl")
"""

from pathlib import Path
from typing import Optional


def step_to_stl(
    input_path: str,
    output_path: str,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> None:
    """
    Convert a STEP file to STL format.

    Args:
        input_path: Path to the input STEP file (.step or .stp)
        output_path: Path for the output STL file (.stl)
        linear_deflection: Controls mesh density - smaller values produce finer meshes
                          (default: 0.1, typical range: 0.01 to 1.0)
        angular_deflection: Angular tolerance in radians for curved surfaces
                           (default: 0.5, typical range: 0.1 to 1.0)

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the input file format is not supported
        RuntimeError: If conversion fails or no suitable library is found
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_file.suffix.lower() not in [".step", ".stp"]:
        raise ValueError(f"Input file must be a STEP file (.step or .stp), got: {input_file.suffix}")
    _convert_with_cadquery(input_file, output_file, linear_deflection, angular_deflection)
    # Try cadquery first (most user-friendly)
    # try:
        
    #     return
    # except ImportError:
    #     pass

    # # Fall back to pythonocc-core
    # try:
    #     _convert_with_pythonocc(input_file, output_file, linear_deflection, angular_deflection)
    #     return
    # except ImportError:
    #     pass

    # raise RuntimeError(
    #     "No suitable CAD library found. Please install one of:\n"
    #     "  - cadquery: pip install cadquery\n"
    #     "  - pythonocc-core: conda install -c conda-forge pythonocc-core"
    # )


def _convert_with_cadquery(
    input_file: Path,
    output_file: Path,
    linear_deflection: float,
    angular_deflection: float,
) -> None:
    """Convert using CadQuery library."""
    import cadquery as cq

    # Import the STEP file
    result = cq.importers.importStep(str(input_file))

    # Export to STL with specified tolerances
    cq.exporters.export(
        result,
        str(output_file),
        exportType="STL",
        tolerance=linear_deflection,
        angularTolerance=angular_deflection,
    )
    print(f"Successfully converted {input_file} to {output_file} using CadQuery")


# def _convert_with_pythonocc(
#     input_file: Path,
#     output_file: Path,
#     linear_deflection: float,
#     angular_deflection: float,
# ) -> None:
#     """Convert using PythonOCC library."""
#     from OCC.Core.STEPControl import STEPControl_Reader
#     from OCC.Core.StlAPI import StlAPI_Writer
#     from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
#     from OCC.Core.IFSelect import IFSelect_RetDone

#     # Read STEP file
#     step_reader = STEPControl_Reader()
#     status = step_reader.ReadFile(str(input_file))

#     if status != IFSelect_RetDone:
#         raise RuntimeError(f"Failed to read STEP file: {input_file}")

#     step_reader.TransferRoots()
#     shape = step_reader.OneShape()

#     if shape.IsNull():
#         raise RuntimeError("No valid shape found in STEP file")

#     # Mesh the shape for STL export
#     mesh = BRepMesh_IncrementalMesh(
#         shape,
#         linear_deflection,
#         False,  # isRelative
#         angular_deflection,
#         True,   # inParallel
#     )
#     mesh.Perform()

#     if not mesh.IsDone():
#         raise RuntimeError("Meshing failed")

#     # Write STL file
#     stl_writer = StlAPI_Writer()
#     stl_writer.SetASCIIMode(False)  # Binary STL (smaller file size)
    
#     success = stl_writer.Write(shape, str(output_file))
    
#     if not success:
#         raise RuntimeError(f"Failed to write STL file: {output_file}")

#     print(f"Successfully converted {input_file} to {output_file} using PythonOCC")


# def step_to_stl_ascii(
#     input_path: str,
#     output_path: str,
#     linear_deflection: float = 0.1,
#     angular_deflection: float = 0.5,
# ) -> None:
#     """
#     Convert a STEP file to ASCII STL format.
    
#     ASCII STL files are human-readable but larger than binary STL.
#     Uses the same parameters as step_to_stl().
#     """
#     input_file = Path(input_path)
#     output_file = Path(output_path)

#     if not input_file.exists():
#         raise FileNotFoundError(f"Input file not found: {input_path}")

#     # Try cadquery first
#     try:
#         import cadquery as cq
#         result = cq.importers.importStep(str(input_file))
#         # CadQuery exports ASCII by default
#         cq.exporters.export(
#             result,
#             str(output_file),
#             exportType="STL",
#             tolerance=linear_deflection,
#             angularTolerance=angular_deflection,
#         )
#         print(f"Successfully converted {input_file} to ASCII STL: {output_file}")
#         return
#     except ImportError:
#         pass

#     # Fall back to pythonocc
#     try:
#         from OCC.Core.STEPControl import STEPControl_Reader
#         from OCC.Core.StlAPI import StlAPI_Writer
#         from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
#         from OCC.Core.IFSelect import IFSelect_RetDone

#         step_reader = STEPControl_Reader()
#         status = step_reader.ReadFile(str(input_file))
#         if status != IFSelect_RetDone:
#             raise RuntimeError(f"Failed to read STEP file: {input_file}")

#         step_reader.TransferRoots()
#         shape = step_reader.OneShape()

#         mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
#         mesh.Perform()

#         stl_writer = StlAPI_Writer()
#         stl_writer.SetASCIIMode(True)  # ASCII mode
#         stl_writer.Write(shape, str(output_file))
#         print(f"Successfully converted {input_file} to ASCII STL: {output_file}")
#         return
#     except ImportError:
#         pass

#     raise RuntimeError("No suitable CAD library found.")


# Command-line interface
if __name__ == "__main__":
    step_to_stl("geometries/Coldspray-heatsink/heatsource.step", "geometries/Coldspray-heatsink/heatsource.stl", linear_deflection=0.005)
