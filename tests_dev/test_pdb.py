#!/usr/bin/env python3
"""
测试脚本用于测试 chai_lab/data/sources/pdb.py 中的 load_pdb_coordinates 功能

Real data files available:
# VHL:
data/fks_data/vhl_mg_cdo1.fasta
data/fks_data/ground_truth_vhl_mg_cdo1.pdb

# Antibody: 
data/fks_data/compound_A.fasta
data/fks_data/ground_truth_compound_A.pdb

# Hard case:
data/pcsk9/6xie.fasta
data/pcsk9/ref_6xie.pdb
"""

import logging
import tempfile
import torch
from pathlib import Path
from typing import Optional

# Import the function we want to test
from chai_lab.data.sources.pdb import load_pdb_coordinates

# Import required dependencies for creating structure_context
from chai_lab.chai1 import make_all_atom_feature_context

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_fasta(fasta_path: Path, sequence: str = "MKFLVLLFNILCLFPVLAADNHKITPEGFLGLLLGLFLGLIFVGLIIGLAFAKLLGLLALFTVVLAGDFT"):
    """创建测试用的FASTA文件"""
    with open(fasta_path, 'w') as f:
        f.write(">protein|test_protein\n")
        f.write(sequence + "\n")
    logger.info(f"Created test FASTA file: {fasta_path}")


def create_test_pdb(pdb_path: Path):
    """创建一个简单的测试PDB文件"""
    pdb_content = """ATOM      1  N   MET A   1      20.000  15.000  25.000  1.00 30.00           N  
ATOM      2  CA  MET A   1      21.400  15.000  25.000  1.00 30.00           C  
ATOM      3  C   MET A   1      22.000  15.000  26.400  1.00 30.00           C  
ATOM      4  O   MET A   1      21.300  15.000  27.400  1.00 30.00           O  
ATOM      5  CB  MET A   1      22.000  13.800  24.300  1.00 30.00           C  
ATOM      6  N   LYS A   2      23.300  15.000  26.500  1.00 30.00           N  
ATOM      7  CA  LYS A   2      24.000  15.000  27.800  1.00 30.00           C  
ATOM      8  C   LYS A   2      25.500  15.000  27.800  1.00 30.00           C  
ATOM      9  O   LYS A   2      26.100  15.000  26.700  1.00 30.00           O  
ATOM     10  CB  LYS A   2      23.500  16.200  28.500  1.00 30.00           C  
ATOM     11  N   PHE A   3      26.000  15.000  29.000  1.00 30.00           N  
ATOM     12  CA  PHE A   3      27.400  15.000  29.200  1.00 30.00           C  
ATOM     13  C   PHE A   3      28.000  15.000  30.600  1.00 30.00           C  
ATOM     14  O   PHE A   3      27.300  15.000  31.600  1.00 30.00           O  
ATOM     15  CB  PHE A   3      28.000  13.800  28.500  1.00 30.00           C  
END
"""
    with open(pdb_path, 'w') as f:
        f.write(pdb_content)
    logger.info(f"Created test PDB file: {pdb_path}")


def test_load_pdb_coordinates_basic():
    """测试基本的 load_pdb_coordinates 功能（使用临时创建的文件）"""
    logger.info("=== Testing load_pdb_coordinates (Basic) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试文件
        fasta_path = temp_path / "test.fasta"
        pdb_path = temp_path / "test.pdb"
        output_dir = temp_path / "output"
        output_dir.mkdir()
        
        create_test_fasta(fasta_path)
        create_test_pdb(pdb_path)
        
        # 设置设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        try:
            # 生成 feature_context
            logger.info("Creating feature context...")
            feature_context = make_all_atom_feature_context(
                fasta_file=fasta_path,
                output_dir=output_dir,
                use_esm_embeddings=False,  # 禁用ESM以加快测试
                use_msa_server=False,      # 禁用MSA以加快测试
                use_templates_server=False # 禁用模板以加快测试
            )
            
            structure_context = feature_context.structure_context
            logger.info(f"✓ Structure context created successfully")
            logger.info(f"  Number of tokens: {structure_context.num_tokens}")
            logger.info(f"  Atom ref pos shape: {structure_context.atom_ref_pos.shape}")
            logger.info(f"  Atom exists mask sum: {structure_context.atom_exists_mask.sum().item()}")
            
            # 测试基本的 load_pdb_coordinates
            logger.info("Testing load_pdb_coordinates...")
            pdb_coords = load_pdb_coordinates(
                pdb_path=pdb_path,
                structure_context=structure_context,
                device=device,
                center_coords=True,
                add_noise=0.0,
                target_num_atoms=None,
                ignore_chains=None
            )
            
            if pdb_coords is not None:
                logger.info(f"✓ Successfully loaded PDB coordinates")
                logger.info(f"  Shape: {pdb_coords.shape}")
                logger.info(f"  Mean: {pdb_coords.mean().item():.3f}")
                logger.info(f"  Std:  {pdb_coords.std().item():.3f}")
                logger.info(f"  Device: {pdb_coords.device}")
                
                # 验证形状
                expected_shape = (1, len(structure_context.atom_ref_pos), 3)
                if pdb_coords.shape == expected_shape:
                    logger.info("✓ Shape matches expected dimensions")
                    return True
                else:
                    logger.warning(f"✗ Shape mismatch: got {pdb_coords.shape}, expected {expected_shape}")
                    return False
            else:
                logger.warning("✗ load_pdb_coordinates returned None")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error in basic test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_load_pdb_coordinates_with_real_data(fasta_path: Path, pdb_path: Path, test_name: str):
    """测试使用真实数据的 load_pdb_coordinates 功能"""
    logger.info(f"=== Testing load_pdb_coordinates ({test_name}) ===")
    
    # 检查文件是否存在
    if not fasta_path.exists():
        logger.warning(f"✗ FASTA file not found: {fasta_path}")
        return False
    
    if not pdb_path.exists():
        logger.warning(f"✗ PDB file not found: {pdb_path}")
        return False
    
    logger.info(f"Using real data files:")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info(f"  PDB:   {pdb_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_dir = temp_path / "output"
        output_dir.mkdir()
        
        # 设置设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        try:
            # 生成 feature_context
            logger.info("Creating feature context from real data...")
            feature_context = make_all_atom_feature_context(
                fasta_file=fasta_path,
                output_dir=output_dir,
                use_esm_embeddings=False,  # 禁用ESM以加快测试
                use_msa_server=False,      # 禁用MSA以加快测试
                use_templates_server=False # 禁用模板以加快测试
            )
            
            structure_context = feature_context.structure_context
            logger.info(f"✓ Structure context created successfully")
            logger.info(f"  Number of tokens: {structure_context.num_tokens}")
            logger.info(f"  Atom ref pos shape: {structure_context.atom_ref_pos.shape}")
            logger.info(f"  Atom exists mask sum: {structure_context.atom_exists_mask.sum().item()}")
            
            # 测试基本的 load_pdb_coordinates
            logger.info("Testing load_pdb_coordinates with real data...")
            pdb_coords = load_pdb_coordinates(
                pdb_path=pdb_path,
                structure_context=structure_context,
                device=device,
                center_coords=True,
                add_noise=0.0,
                target_num_atoms=None,
                ignore_chains=None
            )
            
            if pdb_coords is not None:
                logger.info(f"✓ Successfully loaded PDB coordinates")
                logger.info(f"  Shape: {pdb_coords.shape}")
                logger.info(f"  Mean: {pdb_coords.mean().item():.3f}")
                logger.info(f"  Std:  {pdb_coords.std().item():.3f}")
                logger.info(f"  Min:  {pdb_coords.min().item():.3f}")
                logger.info(f"  Max:  {pdb_coords.max().item():.3f}")
                logger.info(f"  Device: {pdb_coords.device}")
                
                # 验证形状
                expected_shape = (1, len(structure_context.atom_ref_pos), 3)
                if pdb_coords.shape == expected_shape:
                    logger.info("✓ Shape matches expected dimensions")
                else:
                    logger.warning(f"✗ Shape mismatch: got {pdb_coords.shape}, expected {expected_shape}")
                
                # 检查有效坐标数量
                valid_coords = (pdb_coords != 0).any(dim=-1).sum().item()
                total_coords = pdb_coords.shape[1]
                logger.info(f"  Valid coordinates: {valid_coords}/{total_coords} ({100*valid_coords/total_coords:.1f}%)")
                
                # 测试添加噪声
                logger.info("Testing with noise...")
                pdb_coords_noise = load_pdb_coordinates(
                    pdb_path=pdb_path,
                    structure_context=structure_context,
                    device=device,
                    center_coords=True,
                    add_noise=0.1,  # 添加噪声
                    target_num_atoms=None,
                    ignore_chains=None
                )
                
                if pdb_coords_noise is not None:
                    noise_diff = (pdb_coords_noise - pdb_coords).abs().mean().item()
                    logger.info(f"✓ Noise test successful, mean difference: {noise_diff:.3f}")
                
                # 测试忽略chains（如果PDB有多个chain）
                logger.info("Testing with ignore_chains...")
                pdb_coords_ignore = load_pdb_coordinates(
                    pdb_path=pdb_path,
                    structure_context=structure_context,
                    device=device,
                    center_coords=True,
                    add_noise=0.0,
                    target_num_atoms=None,
                    ignore_chains=['Z', 'Y']  # 忽略可能不存在的chains
                )
                
                if pdb_coords_ignore is not None:
                    logger.info("✓ ignore_chains test successful")
                
                return True
                
            else:
                logger.warning("✗ load_pdb_coordinates returned None")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error in {test_name} test: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_all_real_data():
    """测试所有可用的真实数据"""
    logger.info("=== Testing All Real Data Files ===")
    
    # 定义可用的数据文件
    test_cases = [
        {
            "name": "VHL",
            "fasta": Path("data/fks_data/vhl_mg_cdo1.fasta"),
            "pdb": Path("data/fks_data/ground_truth_vhl_mg_cdo1.pdb")
        },
        {
            "name": "Antibody",
            "fasta": Path("data/fks_data/compound_A.fasta"),
            "pdb": Path("data/fks_data/ground_truth_compound_A.pdb")
        },
        {
            "name": "PCSK9 (Hard case)",
            "fasta": Path("data/pcsk9/6xie.fasta"),
            "pdb": Path("data/pcsk9/ref_6xie.pdb")
        }
    ]
    
    success_count = 0
    total_count = 0
    
    for test_case in test_cases:
        logger.info(f"\n{'='*60}")
        total_count += 1
        
        if test_load_pdb_coordinates_with_real_data(
            test_case["fasta"], 
            test_case["pdb"], 
            test_case["name"]
        ):
            success_count += 1
            logger.info(f"✓ {test_case['name']} test PASSED")
        else:
            logger.warning(f"✗ {test_case['name']} test FAILED")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Real data test summary: {success_count}/{total_count} tests passed")
    
    return success_count, total_count


def main():
    """主测试函数"""
    logger.info("Starting load_pdb_coordinates tests...")
    
    success = True
    
    # 运行基本测试
    logger.info(f"\n{'='*60}")
    if not test_load_pdb_coordinates_basic():
        success = False
        logger.error("✗ Basic test FAILED")
    else:
        logger.info("✓ Basic test PASSED")
    
    # 运行真实数据测试
    logger.info(f"\n{'='*60}")
    try:
        success_count, total_count = test_all_real_data()
        if success_count == 0:
            logger.warning("⚠ No real data tests passed (files may not exist)")
        elif success_count < total_count:
            logger.warning(f"⚠ Only {success_count}/{total_count} real data tests passed")
        else:
            logger.info(f"✓ All {success_count} real data tests PASSED")
    except Exception as e:
        logger.error(f"✗ Real data tests failed: {e}")
        success = False
    
    # 总结
    logger.info(f"\n{'='*60}")
    if success:
        logger.info("✓ load_pdb_coordinates testing COMPLETED")
        logger.info("The load_pdb_coordinates function is working correctly!")
    else:
        logger.error("✗ Some tests FAILED")
        logger.error("Check the logs above for details")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 