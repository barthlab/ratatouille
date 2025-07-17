import pytest
import warnings
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.structure.hierarchical_data_structure import (
    Node, Trial, CellSession, Session, Cell, Fov, Mice, Cohort, DataSet
)
from kitchen.structure.meta_data_structure import (
    ObjectUID, TemporalUID, TemporalObjectCoordinate
)
from kitchen.structure.neural_data_structure import NeuralData


class TestNode:
    """Test cases for Node base class."""
    
    def test_valid_creation(self):
        """Test creating a valid Node."""
        obj_uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        temp_uid = TemporalUID(template_id="Template1")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        data = NeuralData()
        
        node = Node(coordinate=coord, data=data)
        assert node.coordinate == coord
        assert node.data == data
        assert node.hash_key == "node"
    
    def test_hash_key_property(self):
        """Test hash_key property returns lowercase class name."""
        obj_uid = ObjectUID(cohort_id="Test")
        temp_uid = TemporalUID(template_id="Test")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        data = NeuralData()
        
        node = Node(coordinate=coord, data=data)
        assert node.hash_key == "node"
    
    def test_equality(self):
        """Test Node equality based on coordinates."""
        obj_uid = ObjectUID(cohort_id="C57BL/6J")
        temp_uid = TemporalUID(template_id="Template1")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        
        node1 = Node(coordinate=coord, data=NeuralData())
        node2 = Node(coordinate=coord, data=NeuralData())  # Different data, same coordinate
        
        assert node1 == node2  # Should be equal based on coordinate
    
    def test_inequality_different_coordinates(self):
        """Test Node inequality with different coordinates."""
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="Template1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="Template2")
        )
        
        node1 = Node(coordinate=coord1, data=NeuralData())
        node2 = Node(coordinate=coord2, data=NeuralData())
        
        assert node1 != node2
    
    def test_inequality_different_types(self):
        """Test Node inequality with different types."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test"),
            temporal_uid=TemporalUID(template_id="Test")
        )
        node = Node(coordinate=coord, data=NeuralData())
        
        assert node != "not a node"
        assert node != 42
        assert node != None
    
    def test_hash(self):
        """Test Node hash based on coordinate."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test"),
            temporal_uid=TemporalUID(template_id="Test")
        )
        
        node1 = Node(coordinate=coord, data=NeuralData())
        node2 = Node(coordinate=coord, data=NeuralData())
        
        assert hash(node1) == hash(node2)
        assert hash(node1) == hash(coord)
    
    def test_repr(self):
        """Test Node string representation."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test"),
            temporal_uid=TemporalUID(template_id="Test")
        )
        node = Node(coordinate=coord, data=NeuralData())
        
        expected = f"node @{coord}"
        assert repr(node) == expected
    
    def test_no_validation_base_node(self):
        """Test that base Node has no UID level validation."""
        # Any coordinate should be valid for base Node
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="Test", day_id=1, session_id=2, chunk_id=3)
        )
        
        # Should not raise any validation errors
        node = Node(coordinate=coord, data=NeuralData())
        assert node.coordinate == coord


class TestSpecializedNodes:
    """Test cases for specialized Node subclasses."""
    
    def test_trial_valid_creation(self):
        """Test creating a valid Trial node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=2, chunk_id=3)
        )
        
        trial = Trial(coordinate=coord, data=NeuralData())
        assert trial.hash_key == "trial"
        assert trial._expected_temporal_uid_level == 'chunk'
        assert trial._expected_object_uid_level == 'cell'
    
    def test_trial_invalid_temporal_level(self):
        """Test Trial with invalid temporal UID level."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=2)  # session level, not chunk
        )
        
        with pytest.raises(TypeError, match="expected temporal_uid @ chunk-level"):
            Trial(coordinate=coord, data=NeuralData())
    
    def test_trial_invalid_object_level(self):
        """Test Trial with invalid object UID level."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1),  # fov level, not cell
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=2, chunk_id=3)
        )
        
        with pytest.raises(TypeError, match="expected object_uid @ cell-level"):
            Trial(coordinate=coord, data=NeuralData())
    
    def test_cell_session_valid_creation(self):
        """Test creating a valid CellSession node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=2)
        )
        
        cell_session = CellSession(coordinate=coord, data=NeuralData())
        assert cell_session.hash_key == "cellsession"
        assert cell_session._expected_temporal_uid_level == 'session'
        assert cell_session._expected_object_uid_level == 'cell'
    
    def test_session_valid_creation(self):
        """Test creating a valid Session node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=2)
        )
        
        session = Session(coordinate=coord, data=NeuralData())
        assert session.hash_key == "session"
        assert session._expected_temporal_uid_level == 'session'
        assert session._expected_object_uid_level == 'fov'
    
    def test_cell_valid_creation(self):
        """Test creating a valid Cell node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cell = Cell(coordinate=coord, data=NeuralData())
        assert cell.hash_key == "cell"
        assert cell._expected_temporal_uid_level == 'template'
        assert cell._expected_object_uid_level == 'cell'
    
    def test_fov_valid_creation(self):
        """Test creating a valid Fov node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        fov = Fov(coordinate=coord, data=NeuralData())
        assert fov.hash_key == "fov"
        assert fov._expected_temporal_uid_level == 'template'
        assert fov._expected_object_uid_level == 'fov'
    
    def test_mice_valid_creation(self):
        """Test creating a valid Mice node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        mice = Mice(coordinate=coord, data=NeuralData())
        assert mice.hash_key == "mice"
        assert mice._expected_temporal_uid_level == 'template'
        assert mice._expected_object_uid_level == 'mice'
    
    def test_cohort_valid_creation(self):
        """Test creating a valid Cohort node."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort = Cohort(coordinate=coord, data=NeuralData())
        assert cohort.hash_key == "cohort"
        assert cohort._expected_temporal_uid_level == 'template'
        assert cohort._expected_object_uid_level == 'cohort'
    
    def test_all_node_types_hierarchy(self):
        """Test that all node types have correct hierarchy expectations."""
        expected_levels = {
            Trial: ('chunk', 'cell'),
            CellSession: ('session', 'cell'),
            Session: ('session', 'fov'),
            Cell: ('template', 'cell'),
            Fov: ('template', 'fov'),
            Mice: ('template', 'mice'),
            Cohort: ('template', 'cohort')
        }
        
        for node_class, (temp_level, obj_level) in expected_levels.items():
            assert node_class._expected_temporal_uid_level == temp_level
            assert node_class._expected_object_uid_level == obj_level


class TestDataSet:
    """Test cases for DataSet class."""
    
    def test_empty_creation(self):
        """Test creating an empty DataSet."""
        dataset = DataSet()
        assert len(dataset.nodes) == 0
        assert len(dataset._fast_lookup) == 0
    
    def test_creation_with_nodes(self):
        """Test creating DataSet with initial nodes."""
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort = Cohort(coordinate=coord1, data=NeuralData())
        mice = Mice(coordinate=coord2, data=NeuralData())
        
        dataset = DataSet(nodes=[cohort, mice])
        assert len(dataset.nodes) == 2
        assert "cohort" in dataset._fast_lookup
        assert "mice" in dataset._fast_lookup
        assert len(dataset._fast_lookup["cohort"]) == 1
        assert len(dataset._fast_lookup["mice"]) == 1
    
    def test_add_single_node(self):
        """Test adding a single node to DataSet."""
        dataset = DataSet()
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        cohort = Cohort(coordinate=coord, data=NeuralData())
        
        dataset.add_node(cohort)
        assert len(dataset.nodes) == 1
        assert cohort in dataset._fast_lookup["cohort"]
    
    def test_add_list_of_nodes(self):
        """Test adding a list of nodes to DataSet."""
        dataset = DataSet()
        
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort = Cohort(coordinate=coord1, data=NeuralData())
        mice = Mice(coordinate=coord2, data=NeuralData())
        
        dataset.add_node([cohort, mice])
        assert len(dataset.nodes) == 2
    
    def test_add_another_dataset(self):
        """Test adding another DataSet to DataSet."""
        dataset1 = DataSet()
        dataset2 = DataSet()
        
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        cohort = Cohort(coordinate=coord, data=NeuralData())
        
        dataset2.add_node(cohort)
        dataset1.add_node(dataset2)
        
        assert len(dataset1.nodes) == 1
        assert cohort in dataset1._fast_lookup["cohort"]
    
    def test_add_duplicate_node_warning(self):
        """Test that adding duplicate nodes raises warning."""
        dataset = DataSet()
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        cohort = Cohort(coordinate=coord, data=NeuralData())
        
        dataset.add_node(cohort)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset.add_node(cohort)  # Add same node again
            
            assert len(w) == 1
            assert "already exists in the dataset" in str(w[0].message)
        
        # Should still have only one node
        assert len(dataset.nodes) == 1
    
    def test_iteration(self):
        """Test iterating over DataSet."""
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort = Cohort(coordinate=coord1, data=NeuralData())
        mice = Mice(coordinate=coord2, data=NeuralData())
        dataset = DataSet(nodes=[cohort, mice])
        
        nodes_list = list(dataset)
        assert len(nodes_list) == 2
        assert cohort in nodes_list
        assert mice in nodes_list
    
    def test_subset(self):
        """Test subset method with criterion function."""
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort = Cohort(coordinate=coord1, data=NeuralData())
        mice = Mice(coordinate=coord2, data=NeuralData())
        dataset = DataSet(nodes=[cohort, mice])
        
        # Get only cohort nodes
        cohort_subset = dataset.subset(lambda node: node.hash_key == "cohort")
        assert len(cohort_subset.nodes) == 1
        assert cohort in cohort_subset.nodes
        assert mice not in cohort_subset.nodes
    
    def test_select(self):
        """Test select method with hash_key and criterion."""
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="BALB/c"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort1 = Cohort(coordinate=coord1, data=NeuralData())
        cohort2 = Cohort(coordinate=coord2, data=NeuralData())
        dataset = DataSet(nodes=[cohort1, cohort2])
        
        # Select cohorts with specific cohort_id
        c57_cohorts = dataset.select("cohort", 
                                   lambda node: node.coordinate.object_uid.cohort_id == "C57BL/6J")
        assert len(c57_cohorts.nodes) == 1
        assert cohort1 in c57_cohorts.nodes
        assert cohort2 not in c57_cohorts.nodes


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for hierarchical data structure."""
    
    def test_node_validation_with_none_expected_levels(self):
        """Test that nodes with None expected levels don't validate."""
        # Create a custom node class with None levels
        class CustomNode(Node):
            _expected_temporal_uid_level = None
            _expected_object_uid_level = None
        
        # Any coordinate should be valid
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="Test", day_id=1)
        )
        
        # Should not raise validation errors
        custom_node = CustomNode(coordinate=coord, data=NeuralData())
        assert custom_node.coordinate == coord
    
    def test_dataset_with_mixed_node_types(self):
        """Test DataSet with mixed node types."""
        coord1 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1")
        )
        
        cohort = Cohort(coordinate=coord1, data=NeuralData())
        cell = Cell(coordinate=coord2, data=NeuralData())
        
        dataset = DataSet(nodes=[cohort, cell])
        assert len(dataset._fast_lookup) == 2
        assert "cohort" in dataset._fast_lookup
        assert "cell" in dataset._fast_lookup
    
    def test_empty_dataset_operations(self):
        """Test operations on empty DataSet."""
        dataset = DataSet()
        
        # Test iteration
        assert list(dataset) == []
        
        # Test subset
        subset = dataset.subset(lambda x: True)
        assert len(subset.nodes) == 0
        
        # Test select
        selected = dataset.select("nonexistent", lambda x: True)
        assert len(selected.nodes) == 0
    
    def test_node_equality_edge_cases(self):
        """Test Node equality edge cases."""
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test"),
            temporal_uid=TemporalUID(template_id="Test")
        )
        
        node = Node(coordinate=coord, data=NeuralData())
        
        # Test equality with None
        assert node != None
        
        # Test equality with different coordinate types
        assert node != coord
        
        # Test NotImplemented return
        result = node.__eq__("string")
        assert result == NotImplemented


class TestDataSetComprehensive:
    """Comprehensive tests for DataSet with diverse node types across the entire hierarchy."""

    def create_full_hierarchy_nodes(self):
        """Create nodes representing the full hierarchy for testing."""
        # Base coordinates for different hierarchy levels
        cohort_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )

        mice_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1"),
            temporal_uid=TemporalUID(template_id="T1")
        )

        fov_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1),
            temporal_uid=TemporalUID(template_id="T1")
        )

        cell_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1")
        )

        session_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=1)
        )

        cell_session_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=1)
        )

        trial_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5),
            temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=1, chunk_id=1)
        )

        # Create all node types
        cohort = Cohort(coordinate=cohort_coord, data=NeuralData())
        mice = Mice(coordinate=mice_coord, data=NeuralData())
        fov = Fov(coordinate=fov_coord, data=NeuralData())
        cell = Cell(coordinate=cell_coord, data=NeuralData())
        session = Session(coordinate=session_coord, data=NeuralData())
        cell_session = CellSession(coordinate=cell_session_coord, data=NeuralData())
        trial = Trial(coordinate=trial_coord, data=NeuralData())

        return {
            'cohort': cohort,
            'mice': mice,
            'fov': fov,
            'cell': cell,
            'session': session,
            'cell_session': cell_session,
            'trial': trial
        }

    def test_mixed_node_type_storage(self):
        """Test DataSet can store different node types simultaneously."""
        nodes = self.create_full_hierarchy_nodes()

        # Create DataSet with all node types
        dataset = DataSet(nodes=list(nodes.values()))

        # Verify all nodes are stored
        assert len(dataset.nodes) == 7

        # Verify all node types are represented in fast lookup
        expected_keys = {'cohort', 'mice', 'fov', 'cell', 'session', 'cellsession', 'trial'}
        assert set(dataset._fast_lookup.keys()) == expected_keys

        # Verify each type has exactly one node
        for key in expected_keys:
            assert len(dataset._fast_lookup[key]) == 1

    def test_hierarchy_validation_in_dataset(self):
        """Test all node types maintain proper coordinate validation when added to DataSet."""
        nodes = self.create_full_hierarchy_nodes()
        dataset = DataSet()

        # Add each node type individually and verify no validation errors
        for node_type, node in nodes.items():
            dataset.add_node(node)
            assert node in dataset.nodes
            assert node in dataset._fast_lookup[node.hash_key]

    def test_fast_lookup_functionality_comprehensive(self):
        """Test _fast_lookup dictionary correctly indexes all node types."""
        nodes = self.create_full_hierarchy_nodes()
        dataset = DataSet(nodes=list(nodes.values()))

        # Test lookup for each node type
        assert dataset._fast_lookup['cohort'][0] == nodes['cohort']
        assert dataset._fast_lookup['mice'][0] == nodes['mice']
        assert dataset._fast_lookup['fov'][0] == nodes['fov']
        assert dataset._fast_lookup['cell'][0] == nodes['cell']
        assert dataset._fast_lookup['session'][0] == nodes['session']
        assert dataset._fast_lookup['cellsession'][0] == nodes['cell_session']
        assert dataset._fast_lookup['trial'][0] == nodes['trial']

        # Test that lookup is consistent with direct access
        for node in dataset.nodes:
            assert node in dataset._fast_lookup[node.hash_key]

    def test_add_node_edge_cases(self):
        """Test edge cases for add_node method."""
        dataset = DataSet()
        nodes = self.create_full_hierarchy_nodes()

        # Test adding empty list
        dataset.add_node([])
        assert len(dataset.nodes) == 0

        # Test adding single node
        dataset.add_node(nodes['cohort'])
        assert len(dataset.nodes) == 1
        assert nodes['cohort'] in dataset._fast_lookup['cohort']

        # Test adding list of nodes
        node_list = [nodes['mice'], nodes['fov']]
        dataset.add_node(node_list)
        assert len(dataset.nodes) == 3
        assert nodes['mice'] in dataset._fast_lookup['mice']
        assert nodes['fov'] in dataset._fast_lookup['fov']

        # Test adding duplicate node (should warn and not add)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset.add_node(nodes['cohort'])
            assert len(w) == 1
            assert "already exists" in str(w[0].message)
        assert len(dataset.nodes) == 3  # Should not increase

        # Test adding invalid types - correctly raise TypeError
        with pytest.raises(TypeError, match="Cannot add object of type.*str.*to DataSet"):
            dataset.add_node("invalid_string")

        with pytest.raises(TypeError, match="Cannot add object of type"):
            dataset.add_node(42)

    def test_string_handling_works_correctly(self):
        """Test that strings are correctly rejected with proper error message."""
        dataset = DataSet()

        # The implementation correctly handles strings by using all() check
        # Strings are sequences, but the all(isinstance(single_node, Node)...) check
        # prevents them from being processed as node sequences

        # Should raise TypeError with clear message
        with pytest.raises(TypeError, match="Cannot add object of type.*str.*to DataSet"):
            dataset.add_node("abc")

        # Same for any string
        with pytest.raises(TypeError, match="Cannot add object of type.*str.*to DataSet"):
            dataset.add_node("single_string")

    def test_sequence_type_checking_works_correctly(self):
        """Test that sequence type checking works correctly for various inputs."""
        dataset = DataSet()

        # The current implementation correctly handles different input types
        # using isinstance(nodes, Sequence) and all(isinstance(single_node, Node)...)

        # Test with various sequence-like objects
        with pytest.raises(TypeError):
            dataset.add_node("string")  # String - correctly raises TypeError

        with pytest.raises(TypeError):
            dataset.add_node(123)  # Integer - correctly raises TypeError

        # Empty string actually works because all() returns True for empty sequences
        dataset.add_node("")  # Empty string works as empty sequence

        # Empty list should work fine (empty sequence of nodes)
        dataset.add_node([])  # Empty list works as empty sequence

    def test_add_node_docstring_placement_issue(self):
        """Test that demonstrates the docstring placement issue in add_node method."""
        # This is more of a code quality issue - docstrings are inside conditional blocks
        # instead of at the method start. This doesn't affect functionality but is poor practice.

        dataset = DataSet()
        nodes = self.create_full_hierarchy_nodes()

        # Test that the method still works despite the docstring placement issue
        dataset.add_node(nodes['cohort'])  # Single node path
        dataset.add_node([nodes['mice']])  # Sequence path
        dataset.add_node(DataSet([nodes['fov']]))  # DataSet path

        assert len(dataset.nodes) == 3

    def test_mixed_types_in_sequence(self):
        """Test handling of mixed types within a sequence."""
        dataset = DataSet()
        nodes = self.create_full_hierarchy_nodes()

        # Test sequence with valid nodes
        valid_sequence = [nodes['cohort'], nodes['mice'], nodes['cell']]
        dataset.add_node(valid_sequence)
        assert len(dataset.nodes) == 3

        # Test sequence with invalid mixed types should fail gracefully
        # Current implementation uses all() check, so mixed sequences fail and get TypeError
        with pytest.raises(TypeError, match="Cannot add object of type.*list.*to DataSet"):
            dataset.add_node([nodes['fov'], "invalid", nodes['trial']])

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time

        # Create many nodes of the same type
        base_coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="C57BL/6J"),
            temporal_uid=TemporalUID(template_id="T1")
        )

        # Create 1000 cohort nodes with different IDs
        large_node_list = []
        for i in range(1000):
            coord = TemporalObjectCoordinate(
                object_uid=ObjectUID(cohort_id=f"Cohort_{i}"),
                temporal_uid=TemporalUID(template_id="T1")
            )
            large_node_list.append(Cohort(coordinate=coord, data=NeuralData()))

        dataset = DataSet()

        # Time bulk addition
        start_time = time.time()
        dataset.add_node(large_node_list)
        end_time = time.time()

        # Verify all nodes were added
        assert len(dataset.nodes) == 1000
        assert len(dataset._fast_lookup['cohort']) == 1000

        # Performance should be reasonable (less than 1 second for 1000 nodes)
        assert (end_time - start_time) < 1.0

        # Test fast lookup performance
        start_time = time.time()
        cohort_nodes = dataset.select('cohort', lambda x: True)
        end_time = time.time()

        assert len(cohort_nodes.nodes) == 1000
        assert (end_time - start_time) < 0.1  # Should be very fast

    def test_cross_hierarchy_operations(self):
        """Test DataSet operations work correctly with mixed node types."""
        nodes = self.create_full_hierarchy_nodes()
        dataset = DataSet(nodes=list(nodes.values()))

        # Test iteration over mixed types
        iterated_nodes = list(dataset)
        assert len(iterated_nodes) == 7
        assert set(iterated_nodes) == set(nodes.values())

        # Test subset with mixed types - get all template-level nodes
        template_nodes = dataset.subset(
            lambda node: hasattr(node, '_expected_temporal_uid_level') and
                        node._expected_temporal_uid_level == 'template'
        )
        expected_template_types = {'cohort', 'mice', 'fov', 'cell'}
        actual_template_types = {node.hash_key for node in template_nodes}
        assert actual_template_types == expected_template_types

        # Test select with specific node type
        cohort_nodes = dataset.select('cohort', lambda _: True)
        assert len(cohort_nodes.nodes) == 1
        assert cohort_nodes.nodes[0] == nodes['cohort']

        # Test select with filtering criteria
        session_level_nodes = dataset.select('session',
                                           lambda node: node.coordinate.temporal_uid.session_id == 1)
        assert len(session_level_nodes.nodes) == 1
        assert session_level_nodes.nodes[0] == nodes['session']

        # Test select with non-existent type
        empty_result = dataset.select('nonexistent', lambda _: True)
        assert len(empty_result.nodes) == 0

    def test_dataset_state_consistency(self):
        """Test that DataSet maintains consistent state during operations."""
        nodes = self.create_full_hierarchy_nodes()
        dataset = DataSet()

        # Add nodes one by one and verify consistency
        for node_name, node in nodes.items():
            dataset.add_node(node)

            # Verify nodes list and fast_lookup are consistent
            assert len(dataset.nodes) == len([n for sublist in dataset._fast_lookup.values() for n in sublist])

            # Verify the specific node is in both structures
            assert node in dataset.nodes
            assert node in dataset._fast_lookup[node.hash_key]

        # Final consistency check
        assert len(dataset.nodes) == 7
        assert len(dataset._fast_lookup) == 7

    def test_coordinate_validation_across_hierarchy(self):
        """Test that coordinate validation works for all hierarchy levels."""
        # Test valid coordinates for each level
        valid_nodes = self.create_full_hierarchy_nodes()
        dataset = DataSet()

        # All should add successfully without validation errors
        for node in valid_nodes.values():
            dataset.add_node(node)

        assert len(dataset.nodes) == 7

        # Test invalid coordinates should raise errors during node creation
        with pytest.raises(TypeError, match="expected.*level"):
            # Wrong temporal level for cohort (should be template, not session)
            invalid_coord = TemporalObjectCoordinate(
                object_uid=ObjectUID(cohort_id="Test"),
                temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=1)
            )
            Cohort(coordinate=invalid_coord, data=NeuralData())

        with pytest.raises(TypeError, match="expected.*level"):
            # Wrong object level for trial (should be cell, not fov)
            invalid_coord = TemporalObjectCoordinate(
                object_uid=ObjectUID(cohort_id="Test", mice_id="M1", fov_id=1),
                temporal_uid=TemporalUID(template_id="T1", day_id=1, session_id=1, chunk_id=1)
            )
            Trial(coordinate=invalid_coord, data=NeuralData())

    def test_string_sequence_handling(self):
        """Test that string sequences are handled correctly (strings are sequences but not node sequences)."""
        dataset = DataSet()

        # Strings are sequences in Python, but should not be treated as node sequences
        # Current implementation correctly raises TypeError due to all() check
        with pytest.raises(TypeError, match="Cannot add object of type.*str.*to DataSet"):
            dataset.add_node("abc")  # Correctly rejected as string

    def test_empty_and_none_handling(self):
        """Test handling of empty sequences and None values."""
        dataset = DataSet()

        # Empty list should work fine
        dataset.add_node([])
        assert len(dataset.nodes) == 0

        # Empty tuple should work fine
        dataset.add_node(())
        assert len(dataset.nodes) == 0

        # None should raise TypeError
        with pytest.raises(TypeError):
            dataset.add_node(None)

    def test_duplicate_detection_comprehensive(self):
        """Test comprehensive duplicate detection across different addition methods."""
        nodes = self.create_full_hierarchy_nodes()
        dataset = DataSet()

        # Add a node
        dataset.add_node(nodes['cohort'])
        assert len(dataset.nodes) == 1

        # Add same node again - should warn and not add
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset.add_node(nodes['cohort'])
            assert len(w) == 1
            assert "already exists" in str(w[0].message)
        assert len(dataset.nodes) == 1

        # Add same node in a list - should still detect duplicate
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset.add_node([nodes['cohort'], nodes['mice']])
            assert len(w) == 1  # Should warn about cohort duplicate
        assert len(dataset.nodes) == 2  # Only mice should be added
        assert nodes['mice'] in dataset.nodes

    def test_node_equality_and_hashing(self):
        """Test that node equality and hashing work correctly for duplicate detection."""
        # Create two nodes with identical coordinates
        coord = TemporalObjectCoordinate(
            object_uid=ObjectUID(cohort_id="Test"),
            temporal_uid=TemporalUID(template_id="Test")
        )

        cohort1 = Cohort(coordinate=coord, data=NeuralData())
        cohort2 = Cohort(coordinate=coord, data=NeuralData())

        # They should be equal and have same hash
        assert cohort1 == cohort2
        assert hash(cohort1) == hash(cohort2)

        # DataSet should treat them as duplicates
        dataset = DataSet()
        dataset.add_node(cohort1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset.add_node(cohort2)
            assert len(w) == 1

        assert len(dataset.nodes) == 1

    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large datasets."""
        import sys

        # Create a large dataset
        large_node_list = []
        for i in range(100):  # Smaller number for memory test
            coord = TemporalObjectCoordinate(
                object_uid=ObjectUID(cohort_id=f"Cohort_{i}"),
                temporal_uid=TemporalUID(template_id="T1")
            )
            large_node_list.append(Cohort(coordinate=coord, data=NeuralData()))

        dataset = DataSet()

        # Measure memory before
        initial_size = sys.getsizeof(dataset.nodes) + sys.getsizeof(dataset._fast_lookup)

        # Add all nodes
        dataset.add_node(large_node_list)

        # Measure memory after
        final_size = sys.getsizeof(dataset.nodes) + sys.getsizeof(dataset._fast_lookup)

        # Verify functionality
        assert len(dataset.nodes) == 100
        assert len(dataset._fast_lookup['cohort']) == 100

        # Memory growth should be reasonable (this is a basic check)
        memory_growth = final_size - initial_size
        assert memory_growth > 0  # Should have grown
        # Memory per node should be reasonable (less than 1KB per node for basic structure)
        assert memory_growth / 100 < 1024


if __name__ == "__main__":
    pytest.main([__file__])
