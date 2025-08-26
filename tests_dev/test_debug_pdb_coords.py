#!/usr/bin/env python3
"""
调试脚本：分析为什么PDB coordinate alignment success rate不是100%
"""
import sys
import torch
from pathlib import Path
sys.path.append(".")

from chai_lab.data.dataset.inference_dataset import load_chains_from_raw, read_inputs
from chai_lab.data.dataset.structure.all_atom_structure_context import AllAtomStructureContext
from chai_lab.data.sources.pdb import load_pdb_coordinates
import logging
from Bio.PDB import PDBParser

# 设置日志级别为DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
# fasta_path = Path("data/fks_data/vhl_mg_cdo1.fasta")
# pdb_path = Path("data/fks_data/ground_truth_vhl_mg_cdo1.pdb")
fasta_path = Path("data/fks_data/vhl_mg_cdo1_lig.fasta")
pdb_path = Path("data/fks_data/ground_truth_vhl_mg_cdo1_lig.pdb")

CENTER_COORDS = False
COPY_COORDS_ONLY = True
ALIGN_SMILES = True


def analyze_pdb_mismatches(pdb_path, structure_context):
    """详细分析PDB坐标匹配的具体失败情况"""
    print(f"\n=== Detailed PDB Mismatch Analysis ===")
    
    # 解析PDB文件
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    
    # 收集PDB中的所有原子
    pdb_atoms = {}
    print("PDB atoms found:")
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            print(f"\nChain {chain_id}:")
            for residue in chain:
                res_name = residue.get_resname()
                res_id = residue.get_id()[1]
                print(f"  Residue {res_name} {res_id}:")
                for atom in residue:
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    pdb_atoms[(chain_id, res_id, res_name, atom_name)] = coord
                    print(f"    {atom_name}: {coord}")
    
    # 分析哪些原子无法匹配
    print(f"\n=== Structure Context Atoms vs PDB ===")
    unmatched_atoms = []
    
    for atom_idx in range(structure_context.num_atoms):
        if not structure_context.atom_exists_mask[atom_idx]:
            continue
            
        atom_name = structure_context.atom_ref_name[atom_idx]
        token_idx = structure_context.atom_token_index[atom_idx].item()
        asym_id = structure_context.token_asym_id[token_idx].item()
        residue_idx = structure_context.token_residue_index[token_idx].item()
        entity_type = structure_context.token_entity_type[token_idx].item()
        element = structure_context.atom_ref_element[atom_idx].item()
        
        from chai_lab.data.parsing.structure.entity_type import EntityType
        entity_type_name = EntityType(entity_type).name
        element_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}.get(element, f'Element_{element}')
        
        # 尝试在PDB中找到这个原子
        found = False
        for (chain_id, res_id, res_name, pdb_atom_name), coords in pdb_atoms.items():
            # 简单的匹配逻辑
            if atom_name == pdb_atom_name:
                found = True
                break
        
        if not found:
            unmatched_atoms.append({
                'atom_idx': atom_idx,
                'atom_name': atom_name,
                'element_symbol': element_symbol,
                'element': element,
                'asym_id': asym_id,
                'residue_idx': residue_idx,
                'entity_type': entity_type_name,
                'token_idx': token_idx
            })
    
    print(f"\nFound {len(unmatched_atoms)} potentially unmatched atoms:")
    for i, atom in enumerate(unmatched_atoms):
        print(f"  {i+1:2d}. {atom['atom_name']:>4s} ({atom['element_symbol']}) - asym_id={atom['asym_id']}, res_idx={atom['residue_idx']}, entity={atom['entity_type']}")
    
    return unmatched_atoms

def analyze_unmatched_atoms():
    """分析具体哪些原子没有找到匹配"""
    
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {fasta_path}")
        return
    
    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}")
        return
    
    # 正确读取FASTA文件
    fasta_inputs = read_inputs(fasta_path, length_limit=None)
    chains = load_chains_from_raw(fasta_inputs)
    
    print(f"\n=== FASTA Chains Analysis ===")
    ligand_smiles = None
    for i, chain in enumerate(chains):
        print(f"Chain {i+1}: {chain.entity_data.entity_type.name}")
        
        # 检查不同的entity data属性
        print(f"  Entity data type: {type(chain.entity_data)}")
        print(f"  Available attributes: {[attr for attr in dir(chain.entity_data) if not attr.startswith('_')]}")
        
        # 正确获取长度和内容
        from chai_lab.data.parsing.structure.entity_type import EntityType
        if chain.entity_data.entity_type == EntityType.LIGAND:
            # For ligands, SMILES is stored in original_record
            ligand_smiles = chain.entity_data.original_record
            print(f"  SMILES: {ligand_smiles}")
        elif hasattr(chain.entity_data, 'full_sequence'):
            length = len(chain.entity_data.full_sequence)
            print(f"  Length: {length} residues")
            print(f"  Sequence: {chain.entity_data.full_sequence[:50]}{'...' if len(chain.entity_data.full_sequence) > 50 else ''}")
        else:
            print(f"  Unknown sequence format")
    
    # 创建structure context (使用正确的方法)
    structure_context = AllAtomStructureContext.merge(
        [c.structure_context for c in chains]
    )
    
    print(f"\n=== Structure Context Analysis ===")
    print(f"Total tokens: {structure_context.num_tokens}")
    print(f"Total atoms: {structure_context.num_atoms}")
    print(f"Existing atoms: {structure_context.atom_exists_mask.sum().item()}")
    
    # 检查氢原子
    print(f"\n=== Atom Element Analysis ===")
    from collections import Counter
    element_counts = Counter(structure_context.atom_ref_element.tolist())
    print("Element distribution:")
    hydrogen_count = 0
    for element, count in sorted(element_counts.items()):
        element_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}.get(element, f'Element_{element}')
        print(f"  {element_symbol} (element {element}): {count} atoms")
    #     if element == 1:  # 氢原子
    #         hydrogen_count = count
    
    # print(f"\n氢原子统计: {hydrogen_count} 个氢原子 / {structure_context.num_atoms} 总原子 = {hydrogen_count/structure_context.num_atoms*100:.1f}%")
    
    # 按asym_id分组分析
    print(f"\n=== Atoms by Asym ID ===")
    for asym_id in torch.unique(structure_context.token_asym_id):
        asym_mask = structure_context.token_asym_id == asym_id
        asym_tokens = asym_mask.sum().item()
        
        # 获取这个asym_id对应的原子
        asym_atom_mask = torch.isin(structure_context.atom_token_index, torch.where(asym_mask)[0])
        asym_atoms = asym_atom_mask.sum().item()
        asym_existing_atoms = (asym_atom_mask & structure_context.atom_exists_mask).sum().item()
        
        entity_type = structure_context.token_entity_type[asym_mask][0].item()
        from chai_lab.data.parsing.structure.entity_type import EntityType
        entity_type_name = EntityType(entity_type).name
        
        print(f"Asym {asym_id.item()}: {entity_type_name}, {asym_tokens} tokens, {asym_atoms} atoms ({asym_existing_atoms} existing)")
        
        # 分析该asym_id的原子元素分布
        asym_elements = structure_context.atom_ref_element[asym_atom_mask]
        asym_element_counts = Counter(asym_elements.tolist())
        asym_hydrogen_count = asym_element_counts.get(1, 0)
        print(f"    氢原子: {asym_hydrogen_count} / {asym_atoms} = {asym_hydrogen_count/asym_atoms*100:.1f}%")
        for element, count in sorted(asym_element_counts.items()):
            element_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}.get(element, f'Element_{element}')
            print(f"    {element_symbol}: {count}")
    
    # 详细分析配体原子
    print(f"\n=== Ligand Atoms Detail ===")
    # 自动找到配体的asym_id
    ligand_asym_id = None
    for asym_id in torch.unique(structure_context.token_asym_id):
        asym_mask = structure_context.token_asym_id == asym_id
        entity_type = structure_context.token_entity_type[asym_mask][0].item()
        from chai_lab.data.parsing.structure.entity_type import EntityType
        if EntityType(entity_type) == EntityType.LIGAND:
            ligand_asym_id = asym_id.item()
            break
    
    if ligand_asym_id is None:
        print("No ligand found in structure context")
        return
    
    ligand_token_mask = structure_context.token_asym_id == ligand_asym_id
    ligand_atom_mask = torch.isin(structure_context.atom_token_index, torch.where(ligand_token_mask)[0])
    
    print(f"Ligand (asym_id={ligand_asym_id}) atoms:")
    ligand_atom_indices = torch.where(ligand_atom_mask)[0]
    for i, atom_idx in enumerate(ligand_atom_indices):
        atom_name = structure_context.atom_ref_name[atom_idx]
        atom_exists = structure_context.atom_exists_mask[atom_idx].item()
        element = structure_context.atom_ref_element[atom_idx].item()
        element_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}.get(element, f'Element_{element}')
        token_idx = structure_context.atom_token_index[atom_idx].item()
        within_token_idx = structure_context.atom_within_token_index[atom_idx].item()
        
        print(f"  Atom {i:2d}: {atom_name:>4s} ({element_symbol}, element {element:2d}, token {token_idx}, within_token {within_token_idx}, exists={atom_exists})")
    
    # 详细分析PDB mismatch
    try:
        unmatched = analyze_pdb_mismatches(pdb_path, structure_context)
    except Exception as e:
        print(f"Error in detailed PDB analysis: {e}")
    
    # 现在尝试加载PDB坐标，使用提取的SMILES字符串
    print(f"\n=== PDB Coordinate Loading with RDKit Alignment ===")
    if ligand_smiles:
        print(f"Using ligand SMILES: {ligand_smiles}")
    else:
        print("Warning: No ligand SMILES found, falling back to element-based matching")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用copy_coords_only模式获取详细的匹配信息
    pdb_coords = load_pdb_coordinates(
        pdb_path=pdb_path,
        structure_context=structure_context,
        device=device,
        center_coords=CENTER_COORDS,
        add_noise=0.0,
        copy_coords_only=COPY_COORDS_ONLY,
        ligand_smiles=ligand_smiles,  # Pass the SMILES string
        align_smiles=ALIGN_SMILES
    )
    
    if pdb_coords is None:
        print("Failed to load PDB coordinates")
        return
    
    print(f"Successfully loaded PDB coordinates: {pdb_coords.shape}")
    
    # 打印PDB和图匹配算法的原子对应关系
    if ligand_smiles:
        print(f"\n=== PDB-SMILES Atom Mapping Analysis ===")
        try:
            from chai_lab.data.sources.pdb import align_pdb_to_smiles_via_substructure_match
            
            # 调用图匹配算法
            aligned_mol, atom_mapping = align_pdb_to_smiles_via_substructure_match(
                smiles_string=ligand_smiles,
                pdb_path=pdb_path,
                ligand_chain_id='E'  # 从PDB分析中看到配体在链E中
            )
            
            if aligned_mol is not None and atom_mapping is not None:
                print(f"Graph isomorphism successful! Found {len(atom_mapping)} atom mappings:")
                
                # 直接使用aligned_mol中的原子信息，而不是重新解析PDB文件
                # 这样确保索引一致性
                
                # 获取SMILES分子的原子信息
                from rdkit import Chem
                smiles_mol = Chem.MolFromSmiles(ligand_smiles)
                smiles_mol = Chem.AddHs(smiles_mol)
                smiles_mol = Chem.RemoveHs(smiles_mol)  # 去除氢原子，与PDB保持一致
                
                print(f"\nAtom mapping details:")
                print(f"{'SMILES_idx':>10} {'SMILES_atom':>12} {'Element':>8} {'->':>4} {'PDB_idx':>8} {'PDB_atom':>10} {'PDB_element':>12}")
                print("-" * 70)
                
                for smiles_idx, pdb_idx in atom_mapping.items():
                    # SMILES原子信息
                    smiles_atom = smiles_mol.GetAtomWithIdx(smiles_idx)
                    smiles_element = smiles_atom.GetSymbol()
                    smiles_name = f"{smiles_element}{smiles_idx+1}"
                    
                    # 使用aligned_mol中的原子信息（这样确保索引一致性）
                    if pdb_idx < aligned_mol.GetNumAtoms():
                        pdb_atom = aligned_mol.GetAtomWithIdx(pdb_idx)
                        pdb_element = pdb_atom.GetSymbol()
                        
                        # 尝试获取原子名称属性
                        if pdb_atom.HasProp('_Name'):
                            pdb_name = pdb_atom.GetProp('_Name')
                        else:
                            pdb_name = f"{pdb_element}{pdb_idx+1}"
                        
                        print(f"{smiles_idx:>10} {smiles_name:>12} {smiles_element:>8} {'->':>4} {pdb_idx:>8} {pdb_name:>10} {pdb_element:>12}")
                    else:
                        print(f"{smiles_idx:>10} {smiles_name:>12} {smiles_element:>8} {'->':>4} {pdb_idx:>8} {'INVALID':>10} {'INVALID':>12}")
                
                # 分析匹配质量
                print(f"\n=== Mapping Quality Analysis ===")
                element_matches = 0
                for smiles_idx, pdb_idx in atom_mapping.items():
                    if pdb_idx < aligned_mol.GetNumAtoms():
                        smiles_atom = smiles_mol.GetAtomWithIdx(smiles_idx)
                        smiles_element = smiles_atom.GetSymbol()
                        pdb_atom = aligned_mol.GetAtomWithIdx(pdb_idx)
                        pdb_element = pdb_atom.GetSymbol()
                        if smiles_element == pdb_element:
                            element_matches += 1
                
                print(f"Element matches: {element_matches}/{len(atom_mapping)} ({element_matches/len(atom_mapping)*100:.1f}%)")
                
                # 详细分析不匹配的原子
                print(f"\n=== Element Mismatch Details ===")
                mismatches = []
                for smiles_idx, pdb_idx in atom_mapping.items():
                    if pdb_idx < aligned_mol.GetNumAtoms():
                        smiles_atom = smiles_mol.GetAtomWithIdx(smiles_idx)
                        smiles_element = smiles_atom.GetSymbol()
                        pdb_atom = aligned_mol.GetAtomWithIdx(pdb_idx)
                        pdb_element = pdb_atom.GetSymbol()
                        if smiles_element != pdb_element:
                            mismatches.append((smiles_idx, pdb_idx, smiles_element, pdb_element))
                
                if mismatches:
                    print(f"Found {len(mismatches)} element mismatches:")
                    for smiles_idx, pdb_idx, smiles_element, pdb_element in mismatches:
                        print(f"  SMILES[{smiles_idx}]={smiles_element} -> PDB[{pdb_idx}]={pdb_element}")
                else:
                    print("No element mismatches found!")
                
                # 打印aligned_mol的详细信息
                print(f"\n=== Aligned Molecule Details ===")
                print(f"Aligned molecule: {aligned_mol.GetNumAtoms()} atoms, {aligned_mol.GetNumBonds()} bonds")
                print(f"Aligned molecule atoms:")
                for i in range(aligned_mol.GetNumAtoms()):
                    atom = aligned_mol.GetAtomWithIdx(i)
                    element = atom.GetSymbol()
                    atom_name = atom.GetProp('_Name') if atom.HasProp('_Name') else f"{element}{i+1}"
                    print(f"  {i:2d}: {atom_name:>4s} ({element})")
                
                # 打印SMILES分子的详细信息
                print(f"\n=== SMILES Molecule Details ===")
                print(f"SMILES molecule: {smiles_mol.GetNumAtoms()} atoms, {smiles_mol.GetNumBonds()} bonds")
                print(f"SMILES molecule atoms:")
                for i in range(smiles_mol.GetNumAtoms()):
                    atom = smiles_mol.GetAtomWithIdx(i)
                    element = atom.GetSymbol()
                    print(f"  {i:2d}: {element}{i+1:>3s} ({element})")
                
                # 打印原始PDB文件的原子信息作为对比
                print(f"\n=== Original PDB File Atom Order (Chain E) ===")
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("structure", pdb_path)
                pdb_atoms_original = []
                for model in structure:
                    for chain in model:
                        if chain.get_id() == 'E':  # 配体链
                            for residue in chain:
                                for atom in residue:
                                    pdb_atoms_original.append({
                                        'name': atom.get_name(),
                                        'element': atom.element,
                                        'coord': atom.get_coord()
                                    })
                
                print(f"Original PDB atoms (Chain E):")
                for i, atom_info in enumerate(pdb_atoms_original):
                    print(f"  {i:2d}: {atom_info['name']:>4s} ({atom_info['element']})")
                
                # 分析RDKit读取的PDB原子顺序
                print(f"\n=== RDKit PDB Molecule Atom Order ===")
                pdb_mol_check = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
                if pdb_mol_check:
                    print(f"RDKit PDB molecule: {pdb_mol_check.GetNumAtoms()} atoms")
                    for i in range(pdb_mol_check.GetNumAtoms()):
                        atom = pdb_mol_check.GetAtomWithIdx(i)
                        element = atom.GetSymbol()
                        atom_name = atom.GetProp('_Name') if atom.HasProp('_Name') else f"{element}{i+1}"
                        print(f"  {i:2d}: {atom_name:>4s} ({element})")
                else:
                    print("Failed to read PDB file with RDKit")
                
                # 打印structure context中的原子名称对比
                print(f"\n=== Structure Context Atom Names ===")
                if len(ligand_atom_indices) > 0:
                    print(f"Structure context ligand atoms:")
                    for i, atom_idx in enumerate(ligand_atom_indices):
                        atom_name = structure_context.atom_ref_name[atom_idx]
                        element = structure_context.atom_ref_element[atom_idx].item()
                        element_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}.get(element, f'Element_{element}')
                        print(f"  {i:2d}: {atom_name:>4s} ({element_symbol})")
                else:
                    print("No ligand atoms in structure context")
                
            else:
                print("Graph isomorphism failed - no atom mapping found")
                
        except Exception as e:
            print(f"Error in graph isomorphism analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # 分析配体对齐质量
    if ligand_smiles and len(ligand_atom_indices) > 0:
        print(f"\n=== Ligand Alignment Quality Analysis ===")
        # 获取配体原子的坐标差异
        ligand_coord_diff = (pdb_coords[0, ligand_atom_indices] - structure_context.atom_ref_pos[ligand_atom_indices].to(device)).abs().sum(dim=-1)
        
        print(f"Ligand coordinate differences:")
        for i, atom_idx in enumerate(ligand_atom_indices):
            atom_name = structure_context.atom_ref_name[atom_idx]
            diff = ligand_coord_diff[i].item()
            print(f"  {atom_name:>4s}: {diff:.2f}Å")
        
        if len(ligand_coord_diff) > 0:
            mean_diff = ligand_coord_diff.mean().item()
            max_diff = ligand_coord_diff.max().item()
            print(f"\nLigand alignment summary:")
            print(f"  Mean difference: {mean_diff:.2f}Å")
            print(f"  Max difference: {max_diff:.2f}Å")
            print(f"  RDKit alignment quality: {'Good' if mean_diff < 5.0 else 'Poor'}")
        else:
            print("No ligand coordinate differences to analyze")
    elif ligand_smiles:
        print(f"\n=== Ligand Alignment Quality Analysis ===")
        print("No ligand atoms found in structure context - cannot analyze alignment quality")
    else:
        print("No ligand SMILES provided - skipping alignment quality analysis")

if __name__ == "__main__":
    analyze_unmatched_atoms() 
