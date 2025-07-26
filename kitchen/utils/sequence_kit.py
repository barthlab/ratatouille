from collections import defaultdict
from typing import Callable, Generator, Iterable, Dict, Any, List, Mapping, Optional, Tuple, TypeVar
import warnings


T = TypeVar('T')
K = TypeVar('K')


def _check(value: T, criterion: Callable[[T], bool] | Tuple[T, ...] | T) -> bool:
    """Helper to check if a value matches a given criterion."""
    if callable(criterion):
        result = criterion(value)
        assert isinstance(result, bool), f"Callable must return bool, but got {type(result)} {result}"
        return result
    if isinstance(criterion, tuple):
        return value in criterion
    return value == criterion

def _matches_criteria(obj: Any, **criteria: Any) -> bool:
    """Check if an object and its attributes match all criteria."""
    # Make a copy to safely modify it.
    criteria = criteria.copy() 
    
    # Check for the '_self' key to match against the object itself.
    if '_self' in criteria:
        self_criterion = criteria.pop('_self')
        if not _check(obj, self_criterion):
            return False

    # Check all remaining criteria against the object's attributes.
    return all(_check(getattr(obj, key, None), val) for key, val in criteria.items())

def find_only_one(datalist: Iterable[T], **criteria: Any) -> T:
    """Find exactly one item matching criteria, or raise an error."""
    matches = [item for item in datalist if _matches_criteria(item, **criteria)]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 match, but found {len(matches)}")
    return matches[0]

def filter_by(datalist: Iterable[T], **criteria: Any) -> list[T]:
    """Filter a list for items matching criteria."""
    return [item for item in datalist if _matches_criteria(item, **criteria)]

def select_from_key(datadict: Dict[K, T], **criteria: Any) -> Dict[K, T]:
    """Select from a dict where keys match criteria."""
    return {k: v for k, v in datadict.items() if _matches_criteria(k, **criteria)}

def select_from_value(datadict: Dict[K, T], **criteria: Any) -> Dict[K, T]:
    """Select from a dict where values match criteria."""
    return {k: v for k, v in datadict.items() if _matches_criteria(v, **criteria)}



def split_by(datalist: Iterable, attr_name: str, _none_warning: bool = True) -> Dict[Any, list]:
    """Split a list into groups based on attribute value."""
    split_dict = defaultdict(list)
    for item in datalist:
        attr_value = getattr(item, attr_name, None)
        if attr_value is None and _none_warning:
            warnings.warn(f"Cannot find attribute {attr_name} in {item}")
        split_dict[attr_value].append(item)
    return split_dict

def group_by(datalist: Iterable[T], key_func: Callable[[T], K], _none_warning: bool = True) -> Dict[K, list[T]]:
    """Group a list into groups based on a key function."""
    group_dict = defaultdict(list)
    for item in datalist:
        key = key_func(item)
        if key is None and _none_warning:
            warnings.warn(f"Cannot find key for {item}")            
        group_dict[key].append(item)
    return group_dict


def zip_dicts(*dcts: Mapping[K, Any]) -> Generator[Tuple[K, Any], None, None]:
    """Find common keys in multiple dicts, and yield (key, (value1, value2, ...)) pairs."""
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield i, tuple(d[i] for d in dcts)


def select_truthy_items(datalist: Iterable[Optional[T]]) -> List[T]:
    """Selects items that are 'truthy'."""
    return [item for item in datalist if item]
