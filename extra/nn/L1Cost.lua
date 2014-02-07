local L1Cost, parent = torch.class('nn.L1Cost','nn.Criterion')

local function mathsign(x) 
    if x==0 then return  2*torch.random(2)-3; end
    if x>0 then return 1; else return -1; end
end

local function numel(x) 
    local dims = x:size():totable() 
    local sz = 1
    for i = 1,#dims do 
        sz = sz * dims[i]
    end
    return sz
end

function L1Cost:__init(weight)
    parent.__init(self)
    --state 
    self.output = torch.Tensor(1) 
    self.gradInput = torch.Tensor() 
    self.weight = weight
    self.sizeAverage = false 
end

function L1Cost:updateOutput(input)

    self.output[1] = input:norm(1)
    self.output:mul(self.weight) 
    
    if self.sizeAverage == true then 
      self.output:div(numel(input)) 
    end 
     
    return self.output 

end

function L1Cost:updateGradInput(input)

    if self.gradOutput == nil then  
    
       self.gradOutput = torch.Tensor(input:size()):typeAs(input):fill(self.weight) 
    
       if self.sizeAverage == true then 
          self.gradOutput = self.gradOutput:div(numel(input)) 
       end
    
    end

    --no need to check this since its L1 
    --we can process whatever we want 
   -- if input:dim() > 2 then
   --    error('input must be vector or matrix')
   -- end
   
    self.gradInput:resize(input:size()):copy(input):apply(mathsign)
    self.gradInput:cmul(self.gradOutput)     
   
   return self.gradInput 

end

--include an empty training functions this enables
--the cost to be added to a network like any other nn.Module 
function L1Cost:zeroGradParameters()

end

function L1Cost:accGradParameters()

end

function L1Cost:updateParameters()

end
