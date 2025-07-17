from typing import Iterable, Dict, Any

def _check(value: Any, criterion: Any) -> bool:
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

def find_only_one(datalist: Iterable, **criteria: Any) -> Any:
    """Find exactly one item matching criteria, or raise an error."""
    matches = [item for item in datalist if _matches_criteria(item, **criteria)]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 match, but found {len(matches)}")
    return matches[0]

def filter_by(datalist: Iterable, **criteria: Any) -> list:
    """Filter a list for items matching criteria."""
    return [item for item in datalist if _matches_criteria(item, **criteria)]

def select_from(datadict: Dict, **criteria: Any) -> Dict:
    """Select from a dict where keys match criteria."""
    return {k: v for k, v in datadict.items() if _matches_criteria(k, **criteria)}