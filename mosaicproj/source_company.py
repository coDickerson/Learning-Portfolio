
class SourceCompany:
    
    def __init__(self, id, requested, dates):
        self.id = id
        self.requested = requested
        self.dates = dates
        self.recommended = []
     
    def __repr__(self):
        return f"Source Company(id={self.id}, requested={self.requested}, dates={self.dates}, recommended={self.recommended})"
