use crate::{optionals::*, utils::*};
use OptionalPair::*;

use hashbrown::{hash_table, DefaultHashBuilder, HashTable, TryReserveError};

use std::{
    borrow::Borrow,
    cell::RefCell,
    default::Default,
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    rc::Rc,
};

#[cfg(doc)]
use hashbrown::HashMap;

// Contains a value and the hash of the item that the value maps to.
pub(crate) struct MappingPair<T> {
    pub(crate) value: T,
    pub(crate) hash: u64,
    pub(crate) id: u64,
}

pub(crate) fn equivalent_key<Q: PartialEq<K>, K>(k: &Q) -> impl '_ + Fn(&MappingPair<K>) -> bool {
    move |x| k.eq(&x.value)
}

pub(crate) fn hash_and_id<Q: PartialEq>(hash: u64, id: u64) -> impl Fn(&MappingPair<Q>) -> bool {
    move |x| id == x.id && hash == x.hash
}

/// Returns an comparision function to be passed to a `HashTable`'s `find_entry` and similar
/// methods. The `id` given to this function will be used by the returned closure and compared
/// against each `MappingPair`'s id.
pub(crate) fn just_id<Q: PartialEq>(id: u64) -> impl Fn(&MappingPair<Q>) -> bool {
    move |x| id == x.id
}

/// A hash map implementation that allows bi-directional, constant time lookups.
///
/// `CycleMap` bijectively maps two sets together. In other words, every item in a set is paired
/// with one and only one item in the other set. It does this while maintaining the same
/// complexitity for lookups as a traditional hash map and while only keeping a single copy of each
/// element.
///
/// `CycleMap` is built on top of the [`HashMap`] implementation found in [`hashbrown`].
/// As such, it uses the same default hashing algorithm as `hashbrown`'s `HashMap`.
///
/// Moreover, the requirements for a "key" for a `HashMap` is required for all values of a
/// CycleMap, namely [`Eq`] and [`Hash`].
///
/// # Examples
/// ```
/// use cycle_map::CycleMap;
///
/// let values = vec![ ("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
/// ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ];
///
/// let mut converter = values.iter()
///                           .cloned()
///                           .map(|(s,i)| (s.to_string(),i))
///                           .collect::<CycleMap<String, u64>>();
///
/// // The map should contain 10 pairs
/// assert_eq!(converter.len(), 10);
///
/// // See if your value number is here
/// if converter.contains_right(&42) {
///     println!( "I know the answer to life!!" );
/// }
///
/// // Get a value from either side!
/// assert_eq!(converter.get_left(&7), Some(&"seven".to_string()));
/// assert_eq!(converter.get_right(&"three".to_string()), Some(&3));
///
/// // Removing an item removes the whole pair
/// assert_eq!(converter.remove_via_right(&7), Some(("seven".to_string(), 7)));
/// assert_eq!(converter.remove_via_left(&"three".to_string()), Some(("three".to_string(), 3)));
///
/// // Remove all pairs with an odd right item
/// converter.retain(|_,i| i % 2 == 0);
/// assert!(converter.contains_right(&2));
///
/// ```
pub struct CycleMap<L, R, St = DefaultHashBuilder> {
    pub(crate) hash_builder: St,
    pub(crate) counter: u64,
    left_set: HashTable<MappingPair<L>>,
    right_set: HashTable<MappingPair<R>>,
}

impl<L, R> CycleMap<L, R, DefaultHashBuilder> {
    #[inline]
    /// Creates a new, empty `CycleMap`.
    /// The capacity of the new map will be 0.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let map: CycleMap<u64, String> = CycleMap::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    /// Creates a new, empty `CycleMap` with inner sets that each have at least the given capacity.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let map: CycleMap<u64, String> = CycleMap::with_capacity(100);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<L, R, S> CycleMap<L, R, S>
where
    L: Eq + Hash,
    R: Eq + Hash,
    S: BuildHasher,
{
    /// Adds a pair of items to the map.
    ///
    /// Should the left element be equal to another left element, the pair containing the old left
    /// item is removed and returned. The same goes for the new right element.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// // Neither 5 nor "5" is in map
    /// let op = map.insert(5, 5.to_string());
    /// assert_eq!(op, Neither);
    ///
    /// // 0 is in the map, its old pairing is removed
    /// let op = map.insert(0, 6.to_string());
    /// assert_eq!(op, SomeLeft((0, 0.to_string())));
    ///
    /// // "1" is in the map, its old pairing is removed
    /// let op = map.insert(7, 1.to_string());
    /// assert_eq!(op, SomeRight((1, 1.to_string())));
    ///
    /// // Both 2 and "3" are in the map, so their old pairings are removed
    /// let op = map.insert(2, 3.to_string());
    /// assert_eq!(op, SomeBoth((2, 2.to_string()),(3, 3.to_string())));
    /// ```
    pub fn insert(&mut self, left: L, right: R) -> InsertOptional<L, R> {
        let opt_from_left = self.remove_via_left(&left);
        let opt_from_right = self.remove_via_right(&right);
        let digest = InsertOptional::from((opt_from_left, opt_from_right));
        let l_hash = make_hash(&self.hash_builder, &left);
        let r_hash = make_hash(&self.hash_builder, &right);
        let left_pairing = MappingPair {
            value: left,
            hash: r_hash,
            id: self.counter,
        };
        let right_pairing = MappingPair {
            value: right,
            hash: l_hash,
            id: self.counter,
        };
        self.counter += 1;
        self.left_set
            .insert_unique(l_hash, left_pairing, make_hasher(&self.hash_builder));
        self.right_set
            .insert_unique(r_hash, right_pairing, make_hasher(&self.hash_builder));
        digest
    }

    /// Returns `true` if both items are in the map and are paired together; otherwise, returns
    /// `false`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert!(map.are_paired(&1, &"1"));
    /// assert!(!map.are_paired(&2, &"1"));
    /// assert!(!map.are_paired(&2, &"2"));
    /// ```
    pub fn are_paired(&self, left: &L, right: &R) -> bool {
        let l_hash = make_hash(&self.hash_builder, left);
        let r_hash = make_hash(&self.hash_builder, right);
        let opt_left = self.left_set.find(l_hash, equivalent_key(left));
        let opt_right = self.right_set.find(r_hash, equivalent_key(right));
        match (opt_left, opt_right) {
            (Some(left), Some(right)) => {
                left.id == right.id && l_hash == right.hash && r_hash == left.hash
            }
            _ => false,
        }
    }

    /// Returns `true` if the item is found and `false` otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert!(map.contains_left(&1));
    /// assert!(!map.contains_left(&2));
    /// ```
    pub fn contains_left(&self, left: &L) -> bool {
        let hash = make_hash(&self.hash_builder, left);
        self.left_set.find(hash, equivalent_key(left)).is_some()
    }

    /// Returns `true` if the item is found and `false` otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert!(map.contains_right(&"1"));
    /// assert!(!map.contains_right(&"2"));
    /// ```
    pub fn contains_right(&self, right: &R) -> bool {
        let hash = make_hash(&self.hash_builder, right);
        self.right_set.find(hash, equivalent_key(right)).is_some()
    }

    /// Removes and returns the give pair from the map provided that they are paired together.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert_eq!(map.remove(&1, &"1"), Some((1, "1")));
    /// assert_eq!(map.remove(&1, &"1"), None);
    /// ```
    pub fn remove(&mut self, left: &L, right: &R) -> Option<(L, R)> {
        if self.are_paired(left, right) {
            self.remove_via_left(left)
        } else {
            None
        }
    }

    /// Finds the item in the left set and its associated item in the right set. Removes both items
    /// and returns them.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert_eq!(map.remove_via_left(&1), Some((1, "1")));
    /// assert_eq!(map.remove_via_left(&1), None);
    /// ```
    pub fn remove_via_left(&mut self, item: &L) -> Option<(L, R)> {
        let l_hash = make_hash(&self.hash_builder, item);
        let left_pairing: MappingPair<L> = self
            .left_set
            .find_entry(l_hash, equivalent_key(item))
            .ok()?
            .remove()
            .0;
        let Ok(entry) = self
            .right_set
            .find_entry(left_pairing.hash, hash_and_id(l_hash, left_pairing.id))
        else {
            unreachable!("TODO");
        };
        let right_pairing = entry.remove().0;
        Some((left_pairing.value, right_pairing.value))
    }

    /// Removes and returns the given item from the right set and its associated item from the left
    /// set.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert_eq!(map.remove_via_right(&"1"), Some((1, "1")));
    /// assert_eq!(map.remove_via_right(&"1"), None);
    /// ```
    pub fn remove_via_right(&mut self, item: &R) -> Option<(L, R)> {
        let r_hash = make_hash(&self.hash_builder, item);
        let right_pairing: MappingPair<R> = self
            .right_set
            .find_entry(r_hash, equivalent_key(item))
            .ok()?
            .remove()
            .0;
        let left_pairing = self
            .left_set
            .find_entry(right_pairing.hash, hash_and_id(r_hash, right_pairing.id))
            // TODO: This should not be the case, and we are silently any error that occurs here.
            // This should be refactored
            .ok()?
            .remove()
            .0;
        Some((left_pairing.value, right_pairing.value))
    }

    /// Removes a pair using the hash of the left item, right item, and their shared pairing id
    fn remove_via_hashes_and_id(&mut self, l_hash: u64, r_hash: u64, id: u64) -> Option<(L, R)> {
        let left_pairing = self
            .left_set
            .find_entry(l_hash, hash_and_id(r_hash, id))
            .ok()?
            .remove()
            .0;
        let right_pairing = self
            .right_set
            .find_entry(r_hash, hash_and_id(l_hash, id))
            // TODO: This should not be the case, and we are silently any error that occurs here.
            // This should be refactored
            .ok()?
            .remove()
            .0;
        Some((left_pairing.value, right_pairing.value))
    }

    /// Swaps an item in the left set with another item, remapping the old item's associated right
    /// item to the new left item, and returns the old left item.
    ///
    /// In other words, the pair `(l1, r)` becomes `(l2, r)`.
    ///
    /// If there is another item in the left set that is equal to the new left item which is paired
    /// to another right item, that cycle is removed.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // 101 is not in the map, so nothing is returned
    /// let op = map.swap_left(&101, 102);
    /// assert_eq!(op, Neither);
    ///
    /// // Swap the 42 in the pair (42, "42") with 101
    /// let op = map.swap_left(&42, 101);
    /// assert_eq!(op, SomeLeft(42));
    /// assert!(map.are_paired(&101, &"42".to_string()));
    ///
    /// // Swap the 84 in the pair (84, "84") with 0 and removes the pair (0, "0")
    /// let op = map.swap_left(&84, 0);
    /// assert_eq!(op, SomeBoth(84, (0, "0".to_string())));
    /// assert!(map.are_paired(&0, &"84".to_string()));
    /// ```
    pub fn swap_left(&mut self, old: &L, new: L) -> OptionalPair<L, (L, R)> {
        // Check for Eq left item and remove that cycle if it exists
        let new_l_hash = make_hash(&self.hash_builder, &new);
        let eq_opt = self.swap_left_eq_check(old, &new, new_l_hash);
        // Find the old left pairing
        let old_l_hash = make_hash(&self.hash_builder, old);
        let Some(l_pairing) = self.left_set.find(old_l_hash, equivalent_key(old)) else {
            return Neither;
        };
        // Use old left pairing to find right pairing
        let r_pairing = self
            .right_set
            .find_mut(l_pairing.hash, hash_and_id(old_l_hash, l_pairing.id))
            .unwrap();
        // Updated right pairing
        r_pairing.hash = new_l_hash;
        // Create new left pairing
        let new_left_pairing = MappingPair {
            value: new,
            hash: l_pairing.hash,
            id: l_pairing.id,
        };
        // Remove old left pairing
        let Ok(entry) = self.left_set.find_entry(old_l_hash, equivalent_key(old)) else {
            panic!()
        };
        let (old_left_item, _) = entry.remove();
        // Insert new left pairing
        self.left_set.insert_unique(
            new_l_hash,
            new_left_pairing,
            make_hasher(&self.hash_builder),
        );
        // Return old left pairing
        OptionalPair::from((Some(old_left_item.value), eq_opt))
    }

    /// Does what [`swap_left`] does, but fails to swap and returns `Neither` if the old item isn't
    /// paired to the given right item.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // Swap the 42 in the pair (42, "42") with 101
    /// let op = map.swap_left_checked(&42, &"42".to_string(), 101);
    /// assert_eq!(op, SomeLeft(42));
    /// assert!(map.are_paired(&101, &"42".to_string()));
    ///
    /// // Fails to swap
    /// let op = map.swap_left_checked(&84, &"85".to_string(), 101);
    /// assert_eq!(op, Neither);
    /// assert!(map.are_paired(&84, &"84".to_string()));
    /// ```
    ///
    /// [`swap_left`]: struct.CycleMap.html#method.swap_left
    pub fn swap_left_checked(&mut self, old: &L, expected: &R, new: L) -> OptionalPair<L, (L, R)> {
        // Check if old and expected are paired
        if !self.are_paired(old, expected) {
            return Neither;
        }
        self.swap_left(old, new)
    }

    /// Does what [`swap_left`] does, but inserts a new pair if the old left item isn't in the map.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // Swap the 42 in the pair (42, "42") with 101
    /// let op = map.swap_left_or_insert(&42, 101, "42".to_string());
    /// assert_eq!(op, SomeLeft(42));
    /// assert!(map.are_paired(&101, &"42".to_string()));
    ///
    /// // Inserts the pair (103, "103") since 102 isn't in the map
    /// let op = map.swap_left_or_insert(&102, 103, "103".to_string());
    /// assert_eq!(op, Neither);
    /// assert!(map.are_paired(&103, &"103".to_string()));
    /// ```
    ///
    /// [`swap_left`]: struct.CycleMap.html#method.swap_left
    pub fn swap_left_or_insert(
        &mut self,
        old: &L,
        new: L,
        to_insert: R,
    ) -> OptionalPair<L, (L, R)> {
        let old_l_hash = make_hash(&self.hash_builder, old);
        if self
            .left_set
            .find(old_l_hash, equivalent_key(old))
            .is_some()
        {
            self.swap_left(old, new)
        } else {
            // TODO: Do further verification on this. All cases _should_ be covered here as this
            // insert should never return a left pair
            self.insert(new, to_insert).map_left(|(l, _)| l)
        }
    }

    /// Pair of the collision checks done in the swap left methods
    fn swap_left_eq_check(&mut self, old: &L, new: &L, new_hash: u64) -> Option<(L, R)> {
        self.left_set.find(new_hash, equivalent_key(new))?;
        if new != old {
            // Remove the problem cycle
            self.remove_via_left(new)
        } else {
            // If old and new are the same, they we are updating an cycle
            None
        }
    }

    /// Swaps an item in the right set with another item, remapping the old item's associated left
    /// item, and returns the old right item
    ///
    /// In other words, the pair `(l, r1)` becomes `(l, r2)`.
    ///
    /// If there is another item in the left set that is equal to the new left item which is paired
    /// to another right item, that cycle is removed.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // 101 is not in the map, so nothing is returned
    /// let op = map.swap_right(&"101".to_string(), "102".to_string());
    /// assert_eq!(op, Neither);
    ///
    /// // Swap the "42" in the pair (42, "42") with "101"
    /// let op = map.swap_right(&"42".to_string(), "101".to_string());
    /// assert_eq!(op, SomeLeft("42".to_string()));
    /// assert!(map.are_paired(&42, &"101".to_string()));
    ///
    /// // Swap the "84" in the pair (84, "84") with "0" and removes the pair (0, "0")
    /// let op = map.swap_right(&"84".to_string(), "0".to_string());
    /// assert_eq!(op, SomeBoth("84".to_string(),(0, "0".to_string())));
    /// assert!(map.are_paired(&84, &"0".to_string()));
    /// ```
    pub fn swap_right(&mut self, old: &R, new: R) -> OptionalPair<R, (L, R)> {
        // Check for Eq left item and remove that cycle if it exists
        let new_r_hash = make_hash(&self.hash_builder, &new);
        let eq_opt = self.swap_right_eq_check(old, &new, new_r_hash);
        // Find the old right pairing
        let old_r_hash = make_hash(&self.hash_builder, old);
        let r_pairing: &MappingPair<R> = match self.right_set.find(old_r_hash, equivalent_key(old))
        {
            Some(p) => p,
            None => {
                return Neither;
            }
        };
        // Use old right pairing to find the left pairing
        let l_pairing: &mut MappingPair<L> = self
            .left_set
            .find_mut(r_pairing.hash, hash_and_id(old_r_hash, r_pairing.id))
            .unwrap();
        // Updated left pairing
        let new_r_hash = make_hash(&self.hash_builder, &new);
        l_pairing.hash = new_r_hash;
        // Create new right pairing
        let new_right_pairing = MappingPair {
            value: new,
            hash: r_pairing.hash,
            id: r_pairing.id,
        };
        // Remove old right pairing
        let Ok(entry) = self.right_set.find_entry(old_r_hash, equivalent_key(old)) else {
            panic!()
        };
        let (old_right_item, _) = entry.remove();

        // Insert new right pairing
        self.right_set.insert_unique(
            new_r_hash,
            new_right_pairing,
            make_hasher(&self.hash_builder),
        );

        // Return old right pairing
        OptionalPair::from((Some(old_right_item.value), eq_opt))
    }

    /// Does what [`swap_right`] does, but fails to swap if the old item isn't paired to the given
    /// left item.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // Swap the "42" in the pair (42, "42") with "101"
    /// let op = map.swap_right_checked(&"42".to_string(), &42, "101".to_string());
    /// assert_eq!(op, SomeLeft("42".to_string()));
    /// assert!(map.are_paired(&42, &"101".to_string()));
    ///
    /// // Fails to swap
    /// let op = map.swap_right_checked(&"84".to_string(), &85, "101".to_string());
    /// assert_eq!(op, Neither);
    /// assert!(map.are_paired(&84, &"84".to_string()));
    /// ```
    ///
    /// [`swap_right`]: struct.CycleMap.html#method.swap_right
    pub fn swap_right_checked(&mut self, old: &R, expected: &L, new: R) -> OptionalPair<R, (L, R)> {
        // Check if old and expected are paired
        if !self.are_paired(expected, old) {
            return Neither;
        } // Things can be removed after this point
        self.swap_right(old, new)
    }

    /// Does what [`swap_right`] does, but inserts a new pair if the old right item isn't in the map.
    ///
    /// Note: This method will never return the `SomeRight` variant of `OptionalPair`.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::{CycleMap, OptionalPair::*};
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // Swap the "42" in the pair (42, "42") with "101"
    /// let op = map.swap_right_or_insert(&"42".to_string(), "101".to_string(), 42);
    /// assert_eq!(op, SomeLeft("42".to_string()));
    /// assert!(map.are_paired(&42, &"101".to_string()));
    ///
    /// // Inserts the pair (103, "103") since 102 isn't in the map
    /// let op = map.swap_right_or_insert(&"102".to_string(), "103".to_string(), 103);
    /// assert_eq!(op, Neither);
    /// assert!(map.are_paired(&103, &"103".to_string()));
    /// ```
    ///
    /// [`swap_right`]: struct.CycleMap.html#method.swap_right
    pub fn swap_right_or_insert(
        &mut self,
        old: &R,
        new: R,
        to_insert: L,
    ) -> OptionalPair<R, (L, R)> {
        // Find the old right pairing
        let old_r_hash = make_hash(&self.hash_builder, old);
        if self
            .right_set
            .find(old_r_hash, equivalent_key(old))
            .is_some()
        {
            self.swap_right(old, new)
        } else {
            // TODO: Do further verification on this. All cases _should_ be covered here
            match self.insert(to_insert, new) {
                InsertOptional::Neither => Neither,
                InsertOptional::SomeRight(pair) => SomeRight(pair),
                _ => {
                    unreachable!("There isn't a left item")
                }
            }
        }
    }

    /// Pair of the collision checks done in the swap left methods
    fn swap_right_eq_check(&mut self, old: &R, new: &R, new_hash: u64) -> Option<(L, R)> {
        self.right_set.find(new_hash, equivalent_key(new))?;
        if new != old {
            // Remove the problem cycle
            self.remove_via_right(new)
        } else {
            // If old and new are the same, they we are updating an cycle
            None
        }
    }

    /// Gets a reference to an item in the left set using an item in the right set.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert_eq!(map.get_left(&"1"), Some(&1));
    /// assert_eq!(map.get_left(&"2"), None);
    /// ```
    pub fn get_left<Q>(&self, item: &Q) -> Option<&L>
    where
        R: Borrow<Q>,
        Q: Hash + Eq + PartialEq<R>,
    {
        let r_hash = make_hash(&self.hash_builder, item);
        let right_pairing: &MappingPair<R> = self.get_right_inner_with_hash(item, r_hash)?;
        match self
            .left_set
            .find(right_pairing.hash, hash_and_id(r_hash, right_pairing.id))
        {
            None => None,
            Some(pairing) => Some(&pairing.value),
        }
    }

    /// Gets a reference to an item in the right set using an item in the left set.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// assert_eq!(map.get_right(&1), Some(&"1"));
    /// assert_eq!(map.get_right(&2), None);
    /// ```
    pub fn get_right<Q>(&self, item: &Q) -> Option<&R>
    where
        L: Borrow<Q>,
        Q: Hash + Eq + PartialEq<L>,
    {
        let l_hash = make_hash(&self.hash_builder, item);
        let left_pairing: &MappingPair<L> = self.get_left_inner_with_hash(item, l_hash)?;
        match self
            .right_set
            .find(left_pairing.hash, hash_and_id(l_hash, left_pairing.id))
        {
            None => None,
            Some(pairing) => Some(&pairing.value),
        }
    }

    /* Might be used in the future
    /// Removes a pair using the hash of the left item, right item, and their shared pairing id
    fn get_via_hashes_and_id(&mut self, l_hash: u64, r_hash: u64, id: u64) -> Option<(&L, &R)> {
    let left_pairing = self.left_set.get(l_hash, hash_and_id(r_hash, id))?;
    let right_pairing = self.right_set.get(r_hash, hash_and_id(l_hash, id)).unwrap();
    Some((&left_pairing.value, &right_pairing.value))
    }
    */

    #[inline]
    fn get_left_inner(&self, item: &L) -> Option<&MappingPair<L>> {
        let hash = make_hash(&self.hash_builder, item);
        self.left_set.find(hash, equivalent_key(item))
    }

    #[inline]
    fn get_left_inner_with_hash<Q>(&self, item: &Q, hash: u64) -> Option<&MappingPair<L>>
    where
        L: Borrow<Q>,
        Q: Hash + Eq + PartialEq<L>,
    {
        self.left_set.find(hash, equivalent_key(item))
    }

    #[inline]
    fn get_right_inner_with_hash<Q>(&self, item: &Q, hash: u64) -> Option<&MappingPair<R>>
    where
        R: Borrow<Q>,
        Q: Hash + Eq + PartialEq<R>,
    {
        self.right_set.find(hash, equivalent_key(item))
    }

    /// Returns an iterator that visits all the pairs in the map in an arbitary order.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let map: CycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for (left, right) in map.iter() {
    ///     println!("left: {left}, right: {right}");
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, L, R, S> {
        Iter {
            left_iter: self.left_set.iter(),
            map_ref: self,
        }
    }

    /// Returns an iterator over items in the left set in an arbitary order.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let map: CycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for left in map.iter_left() {
    ///     println!("left: {left}");
    /// }
    /// ```
    pub fn iter_left(&self) -> SingleIter<'_, L> {
        SingleIter {
            iter: self.left_set.iter(),
        }
    }

    /// Returns an iterator over items in the right set in an arbitary order.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let map: CycleMap<u64, String> = (0..5).map(|i| (i, i.to_string())).collect();
    ///
    /// for right in map.iter_right() {
    ///     println!("right: {right}");
    /// }
    /// ```
    pub fn iter_right(&self) -> SingleIter<'_, R> {
        SingleIter {
            iter: self.right_set.iter(),
        }
    }

    /// Clears the map, returning all pairs as an iterator while keeping the backing memory
    /// allocated for reuse. If the returned iterator is dropped before being fully consumed, it
    /// drops the remaining pairs.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// map.insert(1, "1");
    /// map.insert(2, "2");
    /// let cap = map.capacity();
    ///
    /// for (l, r) in map.drain().take(1) {
    ///     assert!(l == 1 || l == 2);
    ///     assert!(r == "1" || r == "2");
    /// }
    ///
    /// assert!(map.is_empty());
    /// assert_eq!(map.capacity(), cap);
    /// ```
    pub fn drain(&mut self) -> DrainIter<'_, L, R> {
        self.counter = 0;
        DrainIter {
            left_iter: self.left_set.drain(),
            right_ref: &mut self.right_set,
        }
    }

    /// Returns an iterator that removes and yields all pairs that evaluate to `true` in the given
    /// closure while keeping the backing memory allocated.
    ///
    /// If the closure returns `false`, or panics, the element remains in the map and will not be
    /// yielded.
    ///
    /// If the iterator is only partially consumed or not consumed at all, each of the remaining
    /// elements will still be subjected to the closure and removed and dropped if it returns true.
    ///
    /// It is unspecified how many more elements will be subjected to the closure if a panic occurs
    /// in the closure, or a panic occurs while dropping an element, or if the `DrainFilter` value
    /// is leaked.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// // Iterate over the map, remove the pairs with an odd left item
    /// for (l, _) in map.drain_filter(|l,_| l % 2 == 1) {
    ///     assert!(l % 2 == 1);
    /// }
    ///
    /// assert_eq!(map.len(), 50);
    /// ```
    // TODO: The RefCell workaround can be removed once the cursor API for the HashTable is
    // stablizd. When this happens, the return type of this function should be a concrete type
    // again. -> DrainFilterIter<'_, L, R, F>
    pub fn drain_filter<'a, F>(&'a mut self, mut f: F) -> impl 'a + FusedIterator<Item = (L, R)>
    where
        F: 'a + FnMut(&L, &R) -> bool,
    {
        // Much to my dismay, we have to use some kind of interior mutablity to smuggle out the
        // right element and then pull that right element out so it can be yielded. Also, this is
        // going to have to be contained within an structure.
        let push_cache = Rc::new(RefCell::new(None));
        let pull_cache = push_cache.clone();
        let right_set = &mut self.right_set;
        self.left_set
            .extract_if(move |left| {
                let Ok(right_entry) = right_set.find_entry(left.hash, just_id(left.id)) else {
                    unreachable!(
                        "You have encountered a bug in the inner `extract_if` of the \
                    drain_filter. This should not happen, but, since you encountered it, please \
                    file a ticket on the CycleMap github repo."
                    )
                };
                if f(&left.value, &right_entry.get().value) {
                    // We should never find a value here.
                    let _ = push_cache.borrow_mut().insert(right_entry.remove().0);
                    true
                } else {
                    false
                }
            })
            .map(move |l| {
                let Some(r) = pull_cache.borrow_mut().take() else {
                    unreachable!(
                        "You have encountered a bug in the RefCell workaround of the \
                    drain_filter. This should not happen, but, since you encountered it, please \
                    file a ticket on the CycleMap github repo."
                    )
                };
                (l.value, r.value)
            })
    }

    /// Drops all pairs that cause the given closure to return `false`. Pairs are visited in an
    /// arbitary order.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map: CycleMap<u64, String> = (0..100).map(|i| (i,i.to_string())).collect();
    ///
    /// //Remove all pairs with an odd left item
    /// map.retain(|l, _| l % 2 == 0);
    /// assert_eq!(map.len(), 50);
    ///
    /// for (l, _) in map.iter() {
    ///     assert!(l % 2 == 0);
    /// }
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&L, &R) -> bool,
    {
        let mut to_drop: Vec<(u64, u64, u64)> = Vec::with_capacity(self.left_set.len());
        for (left, right) in self.iter() {
            if !f(left, right) {
                let l_hash = make_hash(&self.hash_builder, left);
                let r_hash = make_hash(&self.hash_builder, right);
                let id = self.get_left_inner(left).unwrap().id;
                to_drop.push((l_hash, r_hash, id));
            }
        }
        for (l_hash, r_hash, id) in to_drop {
            self.remove_via_hashes_and_id(l_hash, r_hash, id);
        }
    }

    /// Shrinks the capacity of the map with a lower limit. It will drop down no lower than the
    /// supplied limit while maintaining the internal rules and possibly leaving some space in
    /// accordance with the resize policy.
    ///
    /// This function does nothing if the current capacity is smaller than the supplied minimum capacity.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map: CycleMap<i32, i32> = CycleMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to(10);
    /// assert!(map.capacity() >= 10);
    /// map.shrink_to(0);
    /// assert!(map.capacity() >= 2);
    /// map.shrink_to(10);
    /// assert!(map.capacity() >= 2);
    /// ```
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.left_set
            .shrink_to(min_capacity, make_hasher(&self.hash_builder));
        self.right_set
            .shrink_to(min_capacity, make_hasher(&self.hash_builder));
    }

    /// Shrinks the capacity of the map as much as possible. It will drop down as much as possible
    /// while maintaining the internal rules and possibly leaving some space in accordance with the
    /// resize policy.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map: CycleMap<i32, i32> = CycleMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to_fit();
    /// assert!(map.capacity() >= 2);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.left_set
            .shrink_to(self.len(), make_hasher(&self.hash_builder));
        self.right_set
            .shrink_to(self.len(), make_hasher(&self.hash_builder));
    }

    /// Reserves capacity for at least additional more elements to be inserted in the HashMap. The
    /// collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    /// Panics if the new allocation size overflows usize.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let mut map: CycleMap<&str, i32> = CycleMap::new();
    /// map.reserve(10);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.left_set
            .reserve(additional, make_hasher(&self.hash_builder));
        self.right_set
            .reserve(additional, make_hasher(&self.hash_builder));
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted in the given
    /// HashMap<K,V>. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Errors
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let mut map: CycleMap<&str, isize> = CycleMap::new();
    /// map.try_reserve(10).expect("why is the test harness OMGing on 10 bytes?");
    /// ```
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.left_set
            .try_reserve(additional, make_hasher(&self.hash_builder))?;
        self.right_set
            .try_reserve(additional, make_hasher(&self.hash_builder))?;
        Ok(())
    }
}

impl<L, R, S> CycleMap<L, R, S> {
    /// Creates a `CycleMap`and that uses the given hasher.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and is designed to allow
    /// `CycleMap`s to be resistant to attacks that cause many collisions and very poor
    /// performance. Setting it manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for the CycleMap to be
    /// useful, see its documentation for details.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = CycleMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, "1");
    /// ```
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            counter: 0,
            left_set: HashTable::new(),
            right_set: HashTable::new(),
        }
    }

    /// Creates a `CycleMap` with inner sets that have at least the specified capacity, and that
    /// uses the given hasher.
    ///
    /// The map will be able to hold at least `capacity` many pairs before reallocating.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and is designed to allow
    /// `CycleMap`s to be resistant to attacks that cause many collisions and very poor
    /// performance. Setting it manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for the CycleMap to be
    /// useful, see its documentation for details.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = CycleMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, "1");
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            hash_builder,
            counter: 0,
            left_set: HashTable::with_capacity(capacity),
            right_set: HashTable::with_capacity(capacity),
        }
    }

    /// Returns a reference to the [`BuildHasher`] used by the map
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of items that the inner sets can hold without reallocation.
    ///
    /// Note: Only one value is returned since both sets always have the same capacity.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    /// let map: CycleMap<u64, String> = CycleMap::with_capacity(100);
    /// assert!(map.capacity() >= 100);
    /// ```
    pub fn capacity(&self) -> usize {
        // The size of the sets is always equal
        self.left_set.capacity()
    }

    /// Returns the number of items in the inner sets.
    ///
    /// Note: Only one number is returned since both sets always have the name size.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// assert_eq!(map.len(), 0);
    /// map.insert(1, "1");
    /// assert_eq!(map.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        // The size of the sets is always equal
        self.left_set.len()
    }

    /// Returns `true` if no items are in the map and `false` otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// assert!(map.is_empty());
    /// map.insert(1, "1");
    /// assert_eq!(map.len(), 1);
    /// assert!(!map.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes all items for the map while keeping the backing memory allocated for reuse.
    ///
    /// # Examples
    /// ```rust
    /// use cycle_map::CycleMap;
    ///
    /// let mut map = CycleMap::new();
    /// assert!(map.is_empty());
    /// map.insert(1, "1");
    /// assert!(!map.is_empty());
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.left_set.clear();
        self.right_set.clear();
    }
}

impl<L, R, S> Clone for CycleMap<L, R, S>
where
    L: Eq + Hash + Clone,
    R: Eq + Hash + Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> Self {
        Self {
            left_set: self.left_set.clone(),
            right_set: self.right_set.clone(),
            counter: self.counter,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<L, R, S> Default for CycleMap<L, R, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(Default::default())
    }
}

impl<L, R, S> fmt::Debug for CycleMap<L, R, S>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_set().entries(self.iter()).finish()
    }
}

impl<L, R, S> PartialEq<CycleMap<L, R, S>> for CycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(l, r)| other.are_paired(l, r))
    }
}

impl<L, R, S> Eq for CycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
}

impl<L, R, S> Extend<(L, R)> for CycleMap<L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (L, R)>>(&mut self, iter: T) {
        for (l, r) in iter {
            self.insert(l, r);
        }
    }
}

impl<L, R> FromIterator<(L, R)> for CycleMap<L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn from_iter<T: IntoIterator<Item = (L, R)>>(iter: T) -> Self {
        let mut digest = CycleMap::default();
        digest.extend(iter);
        digest
    }
}

/// An iterator over the entry pairs of a `CycleMap`.
pub struct Iter<'a, L, R, S> {
    left_iter: hash_table::Iter<'a, MappingPair<L>>,
    map_ref: &'a CycleMap<L, R, S>,
}

impl<L, R, S> Clone for Iter<'_, L, R, S> {
    fn clone(&self) -> Self {
        Self {
            left_iter: self.left_iter.clone(),
            map_ref: self.map_ref,
        }
    }
}

impl<L, R, S> fmt::Debug for Iter<'_, L, R, S>
where
    L: Hash + Eq + fmt::Debug,
    R: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, L, R, S> Iterator for Iter<'a, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    type Item = (&'a L, &'a R);

    fn next(&mut self) -> Option<Self::Item> {
        let left_item = self.left_iter.next()?;
        let right_item = self.map_ref.get_right(&left_item.value).unwrap();
        Some((&left_item.value, right_item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left_iter.size_hint()
    }
}

impl<L, R, S> ExactSizeIterator for Iter<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
    fn len(&self) -> usize {
        self.left_iter.len()
    }
}

impl<L, R, S> FusedIterator for Iter<'_, L, R, S>
where
    L: Hash + Eq,
    R: Hash + Eq,
    S: BuildHasher,
{
}

/// An iterator over the left elements of a `CycleMap`.
pub struct SingleIter<'a, T> {
    iter: hash_table::Iter<'a, MappingPair<T>>,
}

impl<T> Clone for SingleIter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<T> fmt::Debug for SingleIter<'_, T>
where
    T: Hash + Eq + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, T> Iterator for SingleIter<'a, T>
where
    T: 'a + Hash + Eq,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| &item.value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> ExactSizeIterator for SingleIter<'_, T>
where
    T: Hash + Eq,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<T> FusedIterator for SingleIter<'_, T> where T: Hash + Eq {}

/// An iterator over the entry pairs of a `CycleMap`.
#[allow(missing_debug_implementations)]
pub struct DrainIter<'a, L, R> {
    left_iter: hash_table::Drain<'a, MappingPair<L>>,
    right_ref: &'a mut HashTable<MappingPair<R>>,
}

impl<L, R> Iterator for DrainIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    type Item = (L, R);

    fn next(&mut self) -> Option<Self::Item> {
        let left = self.left_iter.next()?;
        let right = self
            .right_ref
            .find_entry(left.hash, just_id(left.id))
            // FIXME: Silent ignoring errors...
            .ok()?
            .remove()
            .0;
        Some((left.value, right.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left_iter.size_hint()
    }
}

impl<L, R> ExactSizeIterator for DrainIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
    fn len(&self) -> usize {
        self.left_iter.len()
    }
}

impl<L, R> FusedIterator for DrainIter<'_, L, R>
where
    L: Hash + Eq,
    R: Hash + Eq,
{
}

impl<T: Hash> Hash for MappingPair<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

impl<T: PartialEq> PartialEq for MappingPair<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id) && self.value.eq(&other.value)
    }
}

impl<T: PartialEq> PartialEq<T> for MappingPair<T> {
    fn eq(&self, other: &T) -> bool {
        self.value.eq(other)
    }
}

impl<T: Eq> Eq for MappingPair<T> {}

impl<T: Clone> Clone for MappingPair<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            hash: self.hash,
            id: self.id,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for MappingPair<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MappingPair {{ value: {:?}, hash: {}, id: {} }}",
            self.value, self.hash, self.id
        )
    }
}
