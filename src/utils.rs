use std::hash::{BuildHasher, Hash};

/// Creates a hash
pub(crate) fn make_hash<T, S>(hash_builder: &S, val: &T) -> u64
where
    T: Hash + ?Sized,
    S: BuildHasher,
{
    hash_builder.hash_one(val)
}

/// Creates a hasher
pub(crate) fn make_hasher<T, S>(hash_builder: &S) -> impl '_ + Fn(&T) -> u64
where
    T: Hash,
    S: BuildHasher,
{
    move |val| make_hash::<T, S>(hash_builder, val)
}
